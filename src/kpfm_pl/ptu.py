"""Reader for PicoQuant PTU (TTTR T3) files.

Loads photon records and exposes them as binned time traces, lifetime
histograms and FLID maps. All public methods return :class:`~kpfm_pl.core.Signal`
or :class:`~kpfm_pl.core.Image` objects (or plain dicts) -- there is no plotting
here; feed the returned arrays into matplotlib yourself.
"""

from __future__ import annotations

import struct
import time as _time
from pathlib import Path
from typing import Any

import numpy as np

from .core import Image, Signal
from .fitting import MultiExpFit
from .utils import inflection_left_bound, left_bound_fraction

__all__ = ["PTU"]

_WRAPAROUND = 65536

# Tag type identifiers from the PTU specification.
_TAG_TYPES = {
    "Empty8": 0xFFFF0008,
    "Bool8": 0x00000008,
    "Int8": 0x10000008,
    "BitSet64": 0x11000008,
    "Color8": 0x12000008,
    "Float8": 0x20000008,
    "TDateTime": 0x21000008,
    "Float8Array": 0x2001FFFF,
    "AnsiString": 0x4001FFFF,
    "WideString": 0x4002FFFF,
    "BinaryBlob": 0xFFFFFFFF,
}
_TAG_TYPES = {k: struct.unpack(">i", v.to_bytes(4, "big"))[0] for k, v in _TAG_TYPES.items()}


class PTU:
    """Load a PicoQuant ``.ptu`` TTTR-T3 file.

    Parameters
    ----------
    filepath :
        Path to the ``.ptu`` file.
    dark_counts :
        Dark counts per second, subtracted as a constant offset during lifetime
        fitting.
    smooth_rep_rate :
        When ``True`` use the rounded laser repetition time for record decoding.
    """

    def __init__(self, filepath: str | Path, dark_counts: float = 0.0,
                 smooth_rep_rate: bool = False) -> None:
        self.filepath = Path(filepath)
        self.dark_counts = float(dark_counts)
        self._tags: dict[str, Any] = {}
        self.marker: dict[int, list[int]] = {}
        self.events = np.empty((0, 3))
        self._tt_cache: dict[str, Any] = {}

        with open(self.filepath, "rb") as fh:
            self._read_header(fh)
            if np.isinf(self.coarse_rep_time):
                print(f"{self.filepath.stem}: continuous-wave data, no trigger present.")
                return
            self._read_records(fh, smooth_rep_rate)

    # ------------------------------------------------------------------ header
    def _read_header(self, fh) -> None:
        if fh.read(8).decode("utf-8").strip("\0") != "PQTTTR":
            raise ValueError(f"{self.filepath} is not a valid PTU file.")
        fh.read(8)  # version, unused

        tags: list[tuple[str, Any]] = []
        while True:
            ident = fh.read(32).decode("utf-8").strip("\0")
            idx = struct.unpack("<i", fh.read(4))[0]
            typ = struct.unpack("<i", fh.read(4))[0]
            name = f"{ident}({idx})" if idx > -1 else ident
            tags.append((name, self._read_tag_value(fh, typ)))
            if ident == "Header_End":
                break

        self._tags = dict(tags)
        self.num_records = self._tags["TTResult_NumberOfRecords"]
        self.coarse_rep_time = self._tags["MeasDesc_GlobalResolution"]
        if np.isinf(self.coarse_rep_time):
            return
        digits = int(-np.floor(np.log10(abs(self.coarse_rep_time))))
        self.rep_time = round(self.coarse_rep_time, digits)
        self.resolution = self._tags["MeasDesc_Resolution"]
        self.acquisition_time = self._tags["MeasDesc_AcquisitionTime"] * 1e-3

    @staticmethod
    def _read_tag_value(fh, typ: int) -> Any:
        T = _TAG_TYPES
        if typ == T["Empty8"]:
            fh.read(8)
            return "<empty>"
        if typ == T["Bool8"]:
            return struct.unpack("<q", fh.read(8))[0] != 0
        if typ in (T["Int8"], T["BitSet64"], T["Color8"], T["Float8Array"], T["BinaryBlob"]):
            val = struct.unpack("<q", fh.read(8))[0]
            if typ == T["BinaryBlob"]:
                fh.seek(val, 1)
            return val
        if typ == T["Float8"]:
            return struct.unpack("<d", fh.read(8))[0]
        if typ == T["TDateTime"]:
            f = struct.unpack("<d", fh.read(8))[0]
            return _time.gmtime(int((f - 25569) * 86400))
        if typ == T["AnsiString"]:
            n = struct.unpack("<q", fh.read(8))[0]
            raw = fh.read(n)
            try:
                return raw.decode("utf-8").strip("\0")
            except UnicodeDecodeError:
                return raw.decode("latin1", "ignore").strip("\0")
        if typ == T["WideString"]:
            n = struct.unpack("<q", fh.read(8))[0]
            return fh.read(n).decode("utf-16le", errors="ignore").strip("\0")
        raise ValueError(f"Unknown PTU tag type: {typ}")

    # ----------------------------------------------------------------- records
    def _read_records(self, fh, smooth_rep_rate: bool) -> None:
        rep_time = self.rep_time if smooth_rep_rate else self.coarse_rep_time
        self.rep_time = rep_time

        n_lt_bins = int(rep_time / self.resolution) + 1
        if n_lt_bins > 4096:
            n_lt_bins = 4096
        self.n_lt_bins = n_lt_bins
        self.lt_bins = np.arange(n_lt_bins)
        self.lt_time = self.lt_bins * self.resolution

        ofl = 0
        markers: list[tuple[int, int]] = []
        events: list[np.ndarray] = []
        for _ in range(self.num_records):
            chunk = fh.read(4)
            if len(chunk) < 4:
                print(f"{self.filepath.stem}: file ended earlier than expected.")
                break
            record = struct.unpack("<I", chunk)[0]
            channel = (record >> 28) & 0xF
            dtime = (record >> 16) & 0xFFF
            nsync = record & 0xFFFF
            if channel == 0xF:  # special record
                if dtime == 0:
                    ofl += _WRAPAROUND
                else:
                    markers.append((dtime, ofl + nsync))
            else:
                if dtime >= len(self.lt_bins):
                    continue
                truesync = ofl + nsync
                events.append((truesync, dtime, truesync * rep_time))

        for ch, sync in markers:
            self.marker.setdefault(int(ch), []).append(int(sync))
        self.events = np.array(events, dtype=float) if events else np.empty((0, 3))
        print(f"{self.filepath.stem}: {len(self.events)} photons, "
              f"{sum(len(v) for v in self.marker.values())} markers.")

    # ------------------------------------------------------------------ public
    @property
    def tags(self) -> dict[str, Any]:
        """All header tags as a ``{name: value}`` dict."""
        return dict(self._tags)

    def get_events(self) -> np.ndarray:
        """Raw photon events, shape ``(N, 3)`` = ``[truesync, dtime, time_s]``."""
        return self.events

    def _build_matrix(self, bintime: float) -> None:
        """Build the (time-bin x lifetime-bin) photon count matrix."""
        n_tt = int(np.ceil(self.acquisition_time / bintime))
        matrix = np.zeros((n_tt, self.n_lt_bins), dtype=np.uint32)
        sync_per_bin = bintime / self.rep_time
        tt_idx = (self.events[:, 0] / sync_per_bin).astype(int)
        lt_idx = self.events[:, 1].astype(int)
        np.add.at(matrix, (tt_idx, lt_idx), 1)
        self._matrix = matrix
        self._bintime = bintime

    def time_trace(self, bintime: float = 50e-3, tstart: float = 0.0,
                   tend: float | None = None, laser_pulse: bool = True) -> Signal:
        """Return the binned photon time trace as a :class:`Signal`.

        Parameters
        ----------
        bintime :
            Width of each time bin in seconds.
        tstart, tend :
            Optional time window. ``tend=None`` uses the full acquisition.
        laser_pulse :
            When ``False`` the non-zero portion of the trace is centred and
            zero-padded, and the rising/falling edges are stored as triggers.
        """
        if getattr(self, "_bintime", None) != bintime:
            self._build_matrix(bintime)
        tend = self.acquisition_time if tend is None else min(tend, self.acquisition_time)
        tstart = max(tstart, 0.0)

        start_bin, end_bin = int(tstart / bintime), int(tend / bintime)
        sliced = self._matrix[start_bin:end_bin, :]
        time = tstart + np.arange(sliced.shape[0]) * bintime
        trace = sliced.sum(axis=1)
        meta: dict[str, Any] = {
            "bintime": bintime,
            "matrix": sliced,
            "average_counts": trace.mean() / bintime if len(trace) else 0.0,
        }

        if not laser_pulse:
            nz = np.nonzero(trace)[0]
            if len(nz):
                core = trace[nz[0]:nz[-1] + 1]
                pad = len(trace) - len(core)
                trace = np.pad(core, (pad // 2, pad - pad // 2))
                nz2 = np.nonzero(trace)[0]
                meta["triggers"] = {
                    "left": {"index": int(nz2[0]), "time": float(time[nz2[0]])},
                    "right": {"index": int(nz2[-1]), "time": float(time[nz2[-1]])},
                }
        if meta.get("triggers"):
            self._tt_cache = {"time": time, "trace": trace, "matrix": sliced, "bintime": bintime, "triggers": meta["triggers"]}
        else:
            self._tt_cache = {"time": time, "trace": trace, "matrix": sliced, "bintime": bintime}
        return Signal(time, trace, label=f"{self.filepath.stem} time trace",
                      units={"x": "s", "y": "counts"}, meta=meta)

    def lifetime_histogram(self, matrix: np.ndarray | None = None) -> Signal:
        """Lifetime histogram (summed over time bins) as a :class:`Signal`."""
        if matrix is None:
            if not self._tt_cache:
                self.time_trace()
            matrix = self._tt_cache["matrix"]
        hist = matrix.sum(axis=0)
        return Signal(self.lt_time, hist, label=f"{self.filepath.stem} lifetime",
                      units={"x": "s", "y": "counts"})

    def get_roi_matrix(self, roi: dict[str, float], bintime: float = 50e-3) -> np.ndarray:
        """Extract the photon count matrix for a selected ROI.

        Parameters
        ----------
        roi :
            Dict with ``x_min``, ``x_max``, ``y_min``, ``y_max`` defining the
            region of interest (time and count ranges).
        bintime :
            Width of each time bin in seconds (must match the time_trace call).

        Returns
        -------
        matrix :
            The (time-bin x lifetime-bin) photon count matrix filtered by the
            ROI's time window and count threshold.
        """
        x_min, x_max = roi["x_min"], roi["x_max"]
        y_min, y_max = roi["y_min"], roi["y_max"]

        # Get time trace for the x window
        tt = self.time_trace(bintime=bintime, tstart=x_min, tend=x_max, laser_pulse=False)

        # Extract matrix and trace from the time trace metadata
        matrix = tt["meta"]["matrix"]
        trace = tt["y"]

        # Apply y threshold to filter matrix rows
        mask = (trace >= y_min) & (trace <= y_max)
        roi_matrix = matrix[mask, :]

        return roi_matrix

    def flid(self, roi: dict[str, float] | None = None, *,
             x_min: float | None = None, x_max: float | None = None,
             y_min: float | None = None, y_max: float | None = None,
             bintime: float = 50e-3, n_exp: int = 1, bins: int = 1,
             mode: str = "fraction", fraction: float = 0.95, smoothing: float = 2.0,
             start: float = 0.0, end: float | None = None, sigma: float = 1.5,
             isolated: bool = False, **fit_kwargs) -> Image | list[Image]:
        """Compute a FLID (Fluorescence-Lifetime-Intensity-Distribution) map.

        Each time bin of the (optionally) ROI-restricted matrix is fitted with
        ``n_exp`` exponentials; the resulting ``(lifetime, counts)`` pairs are
        binned into a 2-D histogram (lifetime on x, counts on y).

        Parameters
        ----------
        roi :
            ROI dict ``{"x_min", "x_max", "y_min", "y_max"}`` (e.g. one box from
            :meth:`InteractiveBoxSelector.get_boxes`). Overrides the explicit
            ``x_*``/``y_*`` keyword bounds when given.
        x_min, x_max, y_min, y_max :
            Manual ROI bounds (time in s, counts on y). Any left as ``None``
            default to the full range. If all are ``None`` (and ``roi`` is
            ``None``) the whole time trace is used.
        bintime :
            Time-bin width in seconds.
        n_exp :
            Number of exponential components per time-bin fit.
        bins :
            Lifetime-axis binning factor (``1`` = native resolution).
        mode, fraction, smoothing, start, end :
            Fit-window left-bound selection, as in :meth:`fit_lifetime`.
        sigma :
            Gaussian smoothing applied to the 2-D histogram (``0`` disables).
        isolated :
            When ``True`` return one :class:`Image` per exponential component;
            otherwise return a single map of the per-bin mean lifetime.
        **fit_kwargs :
            Forwarded to :class:`~kpfm_pl.fitting.MultiExpFit`.

        Returns
        -------
        Image or list[Image]
            FLID map(s) with the lifetime axis as ``x`` (ns), counts as ``y``,
            and occurrence as ``data``. Per-bin ``taus``/``counts`` are stored in
            ``meta``.
        """
        from scipy import ndimage

        # Resolve ROI bounds (roi dict takes precedence over explicit kwargs).
        if roi is not None:
            x_min, x_max = roi.get("x_min"), roi.get("x_max")
            y_min, y_max = roi.get("y_min"), roi.get("y_max")

        if x_min is None and x_max is None and y_min is None and y_max is None:
            tt = self.time_trace(bintime=bintime)
            matrix = tt["meta"]["matrix"]
        else:
            xmn = 0.0 if x_min is None else x_min
            xmx = self.acquisition_time if x_max is None else x_max
            tt = self.time_trace(bintime=bintime, tstart=xmn, tend=xmx)
            trace = tt["y"]
            full_matrix = tt["meta"]["matrix"]
            ymn = -np.inf if y_min is None else y_min
            ymx = np.inf if y_max is None else y_max
            matrix = full_matrix[(trace >= ymn) & (trace < ymx), :]

        # Lifetime axis, peak-shifted to ns, and the fit-window mask.
        lt_ns = self.lt_time * 1e9
        lt_roi = matrix.sum(axis=0)
        t0_ns = lt_ns[int(np.argmax(lt_roi))]
        lt_tshift = lt_ns - t0_ns

        if mode == "fraction":
            lb = left_bound_fraction(lt_roi, fraction, smoothing)
            mask = lt_tshift >= lt_tshift[lb]
        elif mode == "inflection":
            lb = inflection_left_bound(lt_roi, smoothing) or 0
            mask = lt_tshift >= lt_tshift[lb]
        elif mode == "manual":
            end_ns = lt_tshift[-1] if end is None else end
            mask = (lt_tshift >= start) & (lt_tshift <= end_ns)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        # Bin the lifetime axis to make per-bin fits more robust.
        binned_lt = lt_ns.copy()
        if bins != 1:
            step = bins * (lt_ns[1] - lt_ns[0])
            binned_lt = np.arange(0.0, np.max(lt_ns), step)
        mask = np.add.reduceat(mask, np.arange(0, mask.shape[0], bins)) > 0
        binned_lt_tshift = binned_lt - t0_ns
        t_fit = binned_lt_tshift[mask]
        binned_matrix = np.add.reduceat(matrix, np.arange(0, matrix.shape[1], bins), axis=1)

        n_binned = len(mask)
        taus: list[np.ndarray] = []
        counts: list[float] = []
        for row in binned_matrix:
            lt_bin = np.asarray(row, dtype=float)
            norm = np.max(lt_bin)
            if not np.isfinite(norm) or norm == 0:
                continue
            lt_bin = lt_bin / norm
            y_fit = lt_bin[mask]
            ymax = np.max(y_fit)
            if not np.isfinite(ymax) or ymax == 0:
                continue
            y_fit = y_fit / ymax
            counts.append(float(np.sum(lt_bin)))
            offset = self.dark_counts * bintime / n_binned / norm
            fit = MultiExpFit(t_fit, y_fit, n_exp, offset, **fit_kwargs)
            if fit.success:
                taus.append(np.asarray(fit.parameters()["tau"], dtype=float))
            else:
                taus.append(np.full(n_exp, np.nan))

        taus = np.array(taus) if taus else np.empty((0, n_exp))
        counts = np.array(counts, dtype=float)
        n_skipped = binned_matrix.shape[0] - len(counts)
        print(f"{self.filepath.stem}: FLID over {len(counts)} time bins "
              f"({n_skipped} skipped).")

        cnt_unit = f"counts/{bintime * 1e3:.0f} ms"
        base_meta = {
            "bintime": bintime, "n_exp": n_exp, "lt_bins": bins, "sigma": sigma,
            "roi": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
            "taus": taus, "counts": counts,
        }

        def _make_image(tau_vals: np.ndarray, label: str) -> Image:
            valid = np.isfinite(tau_vals) & np.isfinite(counts)
            n_hist = max(int(np.sum(valid)), 1)
            hist, tau_edges, cnt_edges = np.histogram2d(
                tau_vals[valid], counts[valid], bins=n_hist)
            if sigma > 0:
                hist = ndimage.gaussian_filter(hist, sigma=sigma)
            tau_centers = 0.5 * (tau_edges[:-1] + tau_edges[1:])
            cnt_centers = 0.5 * (cnt_edges[:-1] + cnt_edges[1:])
            # hist is (n_tau, n_counts); Image data is (ny, nx) = (counts, tau).
            return Image(hist.T, x=tau_centers, y=cnt_centers, label=label,
                         units={"x": "ns", "y": cnt_unit, "z": "occurrence"},
                         meta=base_meta)

        if isolated:
            return [
                _make_image(taus[:, i], f"{self.filepath.stem} FLID (tau {i + 1})")
                for i in range(n_exp)
            ]
        taus_mean = taus.mean(axis=1) if taus.size else np.array([])
        return _make_image(taus_mean, f"{self.filepath.stem} FLID")

    def threshold(self, off: float = 0.0, on: float = 1.0) -> dict[str, Signal]:
        """Split the lifetime histogram into ``on``/``off`` states by count level.

        ``on`` and ``off`` are fractions of the time-trace maximum. Returns a dict
        with ``"on"`` and ``"off"`` :class:`Signal` histograms.
        """
        if not self._tt_cache:
            self.time_trace()
        trace = self._tt_cache["trace"]
        matrix = self._tt_cache["matrix"]
        max_counts = np.max(trace)
        if off == 0 and on == 1:
            on_matrix, off_matrix = matrix, np.zeros_like(matrix)
        else:
            on_matrix = matrix[trace >= int(on * max_counts), :]
            off_matrix = matrix[trace < int(off * max_counts), :]
        return {
            "on": Signal(self.lt_time, on_matrix.sum(axis=0),
                         label=f"on ({on:.0%})", units={"x": "s", "y": "counts"},
                         meta={"fraction": on}),
            "off": Signal(self.lt_time, off_matrix.sum(axis=0),
                          label=f"off ({off:.0%})", units={"x": "s", "y": "counts"},
                          meta={"fraction": off}),
        }

    def fit_lifetime(self, histogram: Signal, n_exp: int = 2, *, mode: str = "fraction",
                     fraction: float = 0.95, smoothing: float = 2.0,
                     start: float = 0.0, end: float | None = None,
                     normalize: bool = False, **fit_kwargs) -> dict[str, Any]:
        """Fit a lifetime ``histogram`` with ``n_exp`` exponentials.

        The fit window left bound is chosen by ``mode``: ``"fraction"`` (rising
        edge), ``"inflection"`` or ``"manual"`` (``start``/``end`` in ns). Returns
        the :meth:`~kpfm_pl.fitting.MultiExpFit.to_dict` payload augmented with the
        time/intensity arrays actually fitted.
        """
        y = np.asarray(histogram["y"], dtype=float)
        t0 = histogram["x"][np.argmax(y)]
        t_ns = (histogram["x"] - t0) * 1e9

        if mode == "fraction":
            lb = left_bound_fraction(y, fraction, smoothing)
            mask = t_ns >= t_ns[lb]
        elif mode == "inflection":
            lb = inflection_left_bound(y, smoothing) or 0
            mask = t_ns >= t_ns[lb]
        elif mode == "manual":
            end = t_ns[-1] if end is None else end
            mask = (t_ns >= start) & (t_ns <= end)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        t_fit, y_fit = t_ns[mask], y[mask]
        norm = 1.0
        if normalize and np.max(y_fit) > 0:
            norm = np.max(y_fit)
            y_fit = y_fit / norm
        offset = self.dark_counts * self._bintime * self._tt_cache["trace"].size \
            / self.n_lt_bins / norm if normalize else \
            self.dark_counts * self._bintime * self._tt_cache["trace"].size / self.n_lt_bins

        fit = MultiExpFit(t_fit, y_fit, n_exp, offset, **fit_kwargs)
        out = fit.to_dict()
        out.update({"t_fit": t_fit, "y_fit": y_fit, "left_bound_ns": float(t_fit[0])})
        return out

    # -------------------------------------------------------------- triggers
    def laser_triggers(self, marker_channel: int = 4) -> dict[str, dict[str, float]] | None:
        """First two laser-trigger markers as ``{"left": ..., "right": ...}``.

        Falls back to the time-trace edges (when ``laser_pulse=False`` was used)
        if no hardware marker is present.
        """
        if marker_channel in self.marker and len(self.marker[marker_channel]) >= 2:
            if not self._tt_cache:
                self.time_trace()
            time = self._tt_cache["time"]
            left_t = self.marker[marker_channel][0] * self.rep_time
            right_t = self.marker[marker_channel][1] * self.rep_time
            triggers = {
                "left": {"index": int(np.argmin(np.abs(time - left_t))), "time": float(left_t)},
                "right": {"index": int(np.argmin(np.abs(time - right_t))), "time": float(right_t)},
            }
        elif self._tt_cache.get("triggers"):
            triggers = self._tt_cache["triggers"]
        else:
            print(f"{self.filepath.stem}: no laser triggers found.")
            return None
        print(f"{self.filepath.stem}: trigger spacing "
              f"{triggers['right']['time'] - triggers['left']['time']:.2f} s")
        return triggers

    def get_markers(self, marker_channel: int = 4) -> np.ndarray | None:
        """All marker times (seconds) on ``marker_channel``."""
        if marker_channel in self.marker:
            return np.array(self.marker[marker_channel]) * self.rep_time
        print(f"{self.filepath.stem}: marker channel {marker_channel} not found.")
        return None
