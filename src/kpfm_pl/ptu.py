"""Reader for PicoQuant PTU (TTTR T3) files.

Loads photon records and exposes them as binned time traces, lifetime
histograms and FLID maps. All public methods return :class:`~kpfm_pl.core.Signal`
or :class:`~kpfm_pl.core.Image` objects (or plain dicts) -- there is no plotting
here; feed the returned arrays into matplotlib yourself.
"""

from __future__ import annotations

import struct
import time as _time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .core import Image, Signal
from .fitting import MultiExpFit
from .utils import (
    TriggerPoint, 
    TriggerPair, 
    TriggerCollection, 
    fit_window_mask, 
    nearest_index, 
    trigger_dict)

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
            self.is_cw = bool(np.isinf(self.coarse_rep_time))
            if self.is_cw:
                # Continuous-wave file: no sync period, so records cannot be
                # decoded into lifetime bins. Keep safe defaults so the object
                # is fully constructed; binning methods raise a clear error.
                self.n_lt_bins = 0
                self.lt_bins = np.array([], dtype=int)
                self.lt_time = np.array([], dtype=float)
                print(f"{self.filepath.stem}: continuous-wave data, no trigger present.")
                return
            self._read_records(fh, smooth_rep_rate)

    # ------------------------------------------------------------------ header
    def _read_header(self, fh) -> None:
        if self._read_exact(fh, 8, "magic").decode("utf-8", errors="replace").strip("\0") != "PQTTTR":
            raise ValueError(f"{self.filepath} is not a valid PTU file.")
        self._read_exact(fh, 8, "version")  # version, unused

        tags: list[tuple[str, Any]] = []
        while True:
            ident = self._read_exact(fh, 32, "tag identifier").decode(
                "utf-8", errors="replace"
            ).strip("\0")
            idx = struct.unpack("<i", self._read_exact(fh, 4, "tag index"))[0]
            typ = struct.unpack("<i", self._read_exact(fh, 4, "tag type"))[0]
            name = f"{ident}({idx})" if idx > -1 else ident
            tags.append((name, self._read_tag_value(fh, typ)))
            if ident == "Header_End":
                break

        self._tags = dict(tags)
        self.num_records = self._tags["TTResult_NumberOfRecords"]
        self.coarse_rep_time = self._tags["MeasDesc_GlobalResolution"]
        if np.isinf(self.coarse_rep_time):
            # Continuous-wave file: keep the infinite rep time but still expose
            # the remaining header attributes so the object is usable.
            self.rep_time = self.coarse_rep_time
        else:
            digits = int(-np.floor(np.log10(abs(self.coarse_rep_time))))
            self.rep_time = round(self.coarse_rep_time, digits)
        self.resolution = self._tags["MeasDesc_Resolution"]
        self.acquisition_time = self._tags["MeasDesc_AcquisitionTime"] * 1e-3

    @staticmethod
    def _read_exact(fh, n: int, field: str) -> bytes:
        """Read exactly ``n`` bytes or reject a truncated PTU field."""
        raw = fh.read(n)
        if len(raw) != n:
            raise ValueError(
                f"Truncated PTU {field}: expected {n} bytes, got {len(raw)}."
            )
        return raw

    @staticmethod
    def _read_length_prefixed(fh) -> bytes:
        """Read a ``<q>`` length prefix and that many bytes, bounded by the file size.

        The length is taken straight from the (potentially untrusted) file, so it
        is validated against the bytes actually remaining before allocating -- a
        corrupt or malicious tag can otherwise request a multi-gigabyte read.
        """
        n = struct.unpack("<q", PTU._read_exact(fh, 8, "payload length"))[0]
        cur = fh.tell()
        end = fh.seek(0, 2)
        fh.seek(cur)
        if n < 0 or n > end - cur:
            raise ValueError(f"Invalid PTU payload length {n} (only {end - cur} bytes left).")
        return PTU._read_exact(fh, n, "payload")

    @staticmethod
    def _read_tag_value(fh, typ: int) -> Any:
        T = _TAG_TYPES
        if typ == T["Empty8"]:
            PTU._read_exact(fh, 8, "empty tag value")
            return "<empty>"
        if typ == T["Bool8"]:
            return struct.unpack("<q", PTU._read_exact(fh, 8, "boolean tag value"))[0] != 0
        if typ in (T["Int8"], T["BitSet64"], T["Color8"]):
            return struct.unpack("<q", PTU._read_exact(fh, 8, "integer tag value"))[0]
        if typ == T["Float8"]:
            return struct.unpack("<d", PTU._read_exact(fh, 8, "float tag value"))[0]
        if typ == T["TDateTime"]:
            f = struct.unpack("<d", PTU._read_exact(fh, 8, "date tag value"))[0]
            return _time.gmtime(int((f - 25569) * 86400))
        if typ == T["Float8Array"]:
            raw = PTU._read_length_prefixed(fh)
            if len(raw) % 8:
                raise ValueError(
                    f"Invalid PTU Float8Array payload length {len(raw)}; expected a multiple of 8."
                )
            return np.frombuffer(raw, dtype="<f8").copy()
        if typ == T["AnsiString"]:
            raw = PTU._read_length_prefixed(fh)
            try:
                return raw.decode("utf-8").strip("\0")
            except UnicodeDecodeError:
                return raw.decode("latin1", "ignore").strip("\0")
        if typ == T["WideString"]:
            return PTU._read_length_prefixed(fh).decode("utf-16le", errors="ignore").strip("\0")
        if typ == T["BinaryBlob"]:
            return PTU._read_length_prefixed(fh)
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
                warnings.warn(
                    f"{self.filepath.stem}: file ended earlier than expected "
                    f"({self.num_records} records declared in header).",
                    RuntimeWarning,
                    stacklevel=2,
                )
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

    def _require_pulsed(self) -> None:
        """Reject time-binned/lifetime analysis on continuous-wave files."""
        if self.is_cw:
            raise ValueError(
                f"{self.filepath.stem}: continuous-wave file (infinite sync "
                "period); time-binned and lifetime analysis require pulsed "
                "(T3) data."
            )

    def _build_matrix(self, bintime: float) -> None:
        """Build the (time-bin x lifetime-bin) photon count matrix."""
        n_tt = int(np.ceil(self.acquisition_time / bintime))
        sync_per_bin = bintime / self.rep_time
        tt_idx = (self.events[:, 0] / sync_per_bin).astype(int)
        # The matrix is sized from the header acquisition time, but tt_idx
        # comes from the photon syncs: with smooth_rep_rate=True (rounded rep
        # time) or files whose records run marginally past the declared
        # duration, the last photons can index past that size. Grow the matrix
        # to fit the data so np.add.at cannot raise a bare IndexError.
        if tt_idx.size:
            n_tt = max(n_tt, int(tt_idx.max()) + 1)
        matrix = np.zeros((n_tt, self.n_lt_bins), dtype=np.uint32)
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
            Refers to the pulse generator in the laser driver (e.g. a 10 MHz
            pulsed laser sends a pulse every 100 ns, needed by the PicoHarp).
            ``True`` (default, and how data is recorded from now on) means the
            pulse generator emits triggers throughout the whole file regardless
            of whether the laser light is on; those triggers are recorded in the
            marker channel and used as the laser triggers.
            ``False`` is a legacy work-around for files where no marker was
            picked up (e.g. the pulse generator only ran while the laser was
            on): the non-zero portion of the trace is centred and zero-padded,
            and its rising/falling edges are stored as triggers. This is bad
            practice and only kept for backwards compatibility.
        """
        self._require_pulsed()
        if not np.isfinite(bintime) or bintime <= 0:
            raise ValueError("bintime must be a positive finite number.")
        if getattr(self, "_bintime", None) != bintime:
            self._build_matrix(bintime)
        tend = self.acquisition_time if tend is None else min(tend, self.acquisition_time)
        tstart = max(tstart, 0.0)

        # Select every acquisition bin that overlaps the requested half-open
        # interval [tstart, tend). Labels remain the actual bin-left times.
        bin_left = np.arange(self._matrix.shape[0], dtype=float) * bintime
        bin_right = np.minimum(bin_left + bintime, self.acquisition_time)
        start_bin = int(np.searchsorted(bin_right, tstart, side="right"))
        end_bin = int(np.searchsorted(bin_left, tend, side="left"))
        end_bin = max(end_bin, start_bin)
        sliced = self._matrix[start_bin:end_bin, :]
        time = bin_left[start_bin:end_bin]
        bin_durations = bin_right[start_bin:end_bin] - time
        # Sum as int64 and convert to float: a uint32 y would silently wrap on
        # downstream subtraction. meta["matrix"] stays integer for memory.
        trace = sliced.sum(axis=1, dtype=np.int64).astype(float)
        selected_duration = float(np.sum(bin_durations))
        meta: dict[str, Any] = {
            "bintime": bintime,
            "matrix": sliced,
            "bin_durations": bin_durations,
            "acquisition_duration": selected_duration,
            "average_counts": trace.sum() / selected_duration if selected_duration else 0.0,
        }

        if not laser_pulse:
            nz = np.nonzero(trace)[0]
            if len(nz):
                core_slice = slice(nz[0], nz[-1] + 1)
                core = trace[core_slice]
                core_matrix = sliced[core_slice, :]
                core_durations = bin_durations[core_slice]
                discarded_durations = np.concatenate(
                    (bin_durations[:nz[0]], bin_durations[nz[-1] + 1:])
                )
                pad = len(trace) - len(core)
                padding = (pad // 2, pad - pad // 2)
                trace = np.pad(core, padding)
                sliced = np.pad(core_matrix, (padding, (0, 0)))
                bin_durations = np.concatenate(
                    (
                        discarded_durations[:padding[0]],
                        core_durations,
                        discarded_durations[padding[0]:],
                    )
                )
                meta["matrix"] = sliced
                meta["bin_durations"] = bin_durations
                nz2 = np.nonzero(trace)[0]
                meta["triggers"] = trigger_dict(
                    nz2[0], time[nz2[0]], nz2[-1], time[nz2[-1]])
        self._tt_cache = {
            "time": time, "trace": trace, "matrix": sliced, "bintime": bintime,
            "bin_durations": bin_durations,
            "acquisition_duration": selected_duration,
            "laser_pulse": bool(laser_pulse),
        }
        if meta.get("triggers"):
            self._tt_cache["triggers"] = meta["triggers"]
        signal = Signal(time, trace, label=f"{self.filepath.stem} time trace",
                        units={"x": "s", "y": "counts"}, meta=meta)
        self._tt_cache["signal"] = signal
        return signal

    def lifetime_histogram(self, matrix: np.ndarray | None = None, *,
                           bintime: float | None = None) -> Signal:
        """Lifetime histogram (summed over time bins) as a :class:`Signal`.

        Pass ``bintime`` with an explicit matrix to retain the acquisition
        duration needed for dark-count correction during fitting.
        """
        self._require_pulsed()
        if matrix is None:
            if not self._tt_cache:
                self.time_trace()
            matrix = self._tt_cache["matrix"]
            bintime = self._tt_cache["bintime"]
            acquisition_duration = self._tt_cache["acquisition_duration"]
        else:
            acquisition_duration = None
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_lt_bins:
            raise ValueError(
                f"matrix must have shape (N, {self.n_lt_bins}); got {matrix.shape}."
            )
        if bintime is not None and (not np.isfinite(bintime) or bintime <= 0):
            raise ValueError("bintime must be a positive finite number.")
        hist = matrix.sum(axis=0)
        meta: dict[str, Any] = {"time_bin_count": int(matrix.shape[0])}
        if bintime is not None:
            meta.update({
                "bintime": float(bintime),
                "acquisition_duration": float(
                    matrix.shape[0] * bintime
                    if acquisition_duration is None else acquisition_duration
                ),
            })
        return Signal(self.lt_time, hist, label=f"{self.filepath.stem} lifetime",
                      units={"x": "s", "y": "counts"}, meta=meta)

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
        tt = self.time_trace(bintime=bintime, tstart=x_min, tend=x_max)

        # Extract matrix and trace from the time trace metadata
        matrix = tt["meta"]["matrix"]
        trace = tt["y"]

        # Apply y threshold to filter matrix rows
        mask = (trace >= y_min) & (trace <= y_max)
        roi_matrix = matrix[mask, :]

        return roi_matrix

    def _fit_time_bin_lifetimes(
        self, roi: dict[str, float] | None = None, *,
        x_min: float | None = None, x_max: float | None = None,
        y_min: float | None = None, y_max: float | None = None,
        bintime: float = 50e-3, n_exp: int = 1, bins: int = 1,
        mode: str = "fraction", fraction: float = 0.95,
        smoothing: float = 2.0, start: float = 0.0,
        end: float | None = None, laser_pulse: bool | None = None,
        **fit_kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | None]]:
        """Fit lifetimes per time bin while preserving time-bin positions."""
        self._require_pulsed()
        if (isinstance(n_exp, (bool, np.bool_))
                or not isinstance(n_exp, (int, np.integer)) or n_exp < 1):
            raise ValueError("n_exp must be a positive integer.")
        if (isinstance(bins, (bool, np.bool_))
                or not isinstance(bins, (int, np.integer)) or bins < 1):
            raise ValueError("bins must be a positive integer.")

        if roi is not None:
            x_min, x_max = roi.get("x_min"), roi.get("x_max")
            y_min, y_max = roi.get("y_min"), roi.get("y_max")

        xmn = 0.0 if x_min is None else x_min
        xmx = self.acquisition_time if x_max is None else x_max
        # Lifetime values must be fitted from the same time-bin layout as the
        # PL trace they accompany.  In particular, ``laser_pulse=False``
        # centres and pads legacy marker-less traces.  Rebuilding with the
        # normal default would silently put the lifetime values back on the
        # unshifted bins.  Unless explicitly overridden, inherit the mode of
        # the most recently generated time trace.
        if laser_pulse is None:
            laser_pulse = bool(self._tt_cache.get("laser_pulse", True))
        elif not isinstance(laser_pulse, (bool, np.bool_)):
            raise TypeError("laser_pulse must be a bool or None.")

        if laser_pulse:
            tt = self.time_trace(
                bintime=bintime, tstart=xmn, tend=xmx, laser_pulse=True
            )
            time = np.asarray(tt["x"], dtype=float)
            counts = np.asarray(tt["y"], dtype=float)
            matrix = np.asarray(tt["meta"]["matrix"])
        else:
            # Apply legacy centring to the complete acquisition before taking
            # an x-window.  Centring only a requested sub-window can move the
            # same photons to different bins than in the full PL trace.
            tt = self.time_trace(bintime=bintime, laser_pulse=False)
            full_time = np.asarray(tt["x"], dtype=float)
            full_counts = np.asarray(tt["y"], dtype=float)
            full_matrix = np.asarray(tt["meta"]["matrix"])
            durations = np.asarray(tt["meta"]["bin_durations"], dtype=float)
            in_window = (full_time + durations > xmn) & (full_time < xmx)
            time = full_time[in_window]
            counts = full_counts[in_window]
            matrix = full_matrix[in_window]

        ymn = -np.inf if y_min is None else y_min
        ymx = np.inf if y_max is None else y_max
        selected = (counts >= ymn) & (counts < ymx)
        selected_matrix = matrix[selected, :]
        if selected_matrix.shape[0] == 0 or not np.any(selected_matrix):
            raise ValueError("The selected lifetime region contains no photons.")

        # Use the selected region to choose one common lifetime fit window, then
        # fit each selected row independently. Unselected/invalid/failed rows
        # remain NaN so their positions stay aligned with ``time``.
        lt_ns = self.lt_time * 1e9
        lt_roi = selected_matrix.sum(axis=0)
        t0_ns = lt_ns[int(np.argmax(lt_roi))]
        lt_tshift = lt_ns - t0_ns
        mask = fit_window_mask(lt_tshift, lt_roi, mode, fraction=fraction,
                               smoothing=smoothing, start=start, end=end)

        group_starts = np.arange(0, matrix.shape[1], bins)
        if group_starts.size < 2:
            raise ValueError("bins must leave at least two lifetime groups for fitting.")
        group_sizes = np.minimum(bins, matrix.shape[1] - group_starts)
        binned_lt = np.add.reduceat(lt_ns, group_starts) / group_sizes
        mask = np.add.reduceat(mask, group_starts) > 0
        t_fit = (binned_lt - t0_ns)[mask]
        if t_fit.size < 2:
            raise ValueError("The selected lifetime fit window contains fewer than two bins.")
        binned_matrix = np.add.reduceat(matrix, group_starts, axis=1)

        taus = np.full((len(time), n_exp), np.nan, dtype=float)
        n_binned = len(mask)
        for index in np.flatnonzero(selected):
            lt_bin = np.asarray(binned_matrix[index], dtype=float)
            norm = np.max(lt_bin)
            if not np.isfinite(norm) or norm == 0:
                continue
            y_fit = lt_bin[mask] / norm
            ymax = np.max(y_fit)
            if not np.isfinite(ymax) or ymax == 0:
                continue
            y_fit = y_fit / ymax
            offset = self.dark_counts * bintime / n_binned / norm
            fit = MultiExpFit(t_fit, y_fit, n_exp, offset, **fit_kwargs)
            if fit.success:
                fitted_taus = np.asarray(fit.parameters()["tau"], dtype=float)
                if fitted_taus.shape == (n_exp,) and np.all(np.isfinite(fitted_taus)):
                    taus[index] = fitted_taus

        bounds = {"x_min": x_min, "x_max": x_max,
                  "y_min": y_min, "y_max": y_max}
        return time, taus, counts, bounds

    def lifetime_trace(self, roi: dict[str, float] | None = None, *,
                       x_min: float | None = None, x_max: float | None = None,
                       y_min: float | None = None, y_max: float | None = None,
                       bintime: float = 50e-3, n_exp: int = 1, bins: int = 1,
                       mode: str = "fraction", fraction: float = 0.95,
                       smoothing: float = 2.0, start: float = 0.0,
                       end: float | None = None,
                       laser_pulse: bool | None = None, **fit_kwargs) -> Signal:
        """Return mean fitted lifetime versus acquisition time.

        The returned ``Signal.x`` contains the time-bin positions and
        ``Signal.y`` contains the arithmetic mean of the fitted exponential
        lifetimes in ns. There is exactly one value per time bin. A value is
        ``NaN`` when the bin is outside the requested intensity ROI, contains
        no usable photons, or its fit fails. Individual fitted components are
        available as ``signal.meta["taus"]`` with shape ``(time bins, n_exp)``.
        By default, the trace inherits the ``laser_pulse`` mode of the most
        recent :meth:`time_trace` call.  Pass the option explicitly when no
        time trace has been generated first.
        """
        time, taus, counts, bounds = self._fit_time_bin_lifetimes(
            roi, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            bintime=bintime, n_exp=n_exp, bins=bins, mode=mode,
            fraction=fraction, smoothing=smoothing, start=start, end=end,
            laser_pulse=laser_pulse, **fit_kwargs,
        )
        mean_taus = np.full(len(time), np.nan, dtype=float)
        valid = np.all(np.isfinite(taus), axis=1)
        mean_taus[valid] = np.mean(taus[valid], axis=1)
        return Signal(
            time, mean_taus,
            label=f"{self.filepath.stem} mean lifetime",
            units={"x": "s", "y": "ns"},
            meta={"bintime": bintime, "n_exp": n_exp, "lt_bins": bins,
                  "roi": bounds, "taus": taus, "counts": counts,
                  "laser_pulse": bool(self._tt_cache["laser_pulse"])},
        )

    def flid(self, roi: dict[str, float] | None = None, *,
             x_min: float | None = None, x_max: float | None = None,
             y_min: float | None = None, y_max: float | None = None,
             bintime: float = 50e-3, n_exp: int = 1, bins: int = 1,
             lifetime_bins: int = 100, intensity_bins: int = 100,
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
        lifetime_bins, intensity_bins :
            Number of histogram bins on the fitted-lifetime and raw-count axes.
            Each must be between 1 and 1024, which keeps map allocation bounded.
            Relevant only for plotting the histogram.
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

        for name, value in (("lifetime_bins", lifetime_bins),
                            ("intensity_bins", intensity_bins)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or not 1 <= value <= 1024):
                raise ValueError(f"{name} must be an integer between 1 and 1024.")

        time, taus, counts, bounds = self._fit_time_bin_lifetimes(
            roi, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            bintime=bintime, n_exp=n_exp, bins=bins, mode=mode,
            fraction=fraction, smoothing=smoothing, start=start, end=end,
            **fit_kwargs,
        )
        n_fitted = int(np.count_nonzero(np.all(np.isfinite(taus), axis=1)))
        print(f"{self.filepath.stem}: FLID fitted {n_fitted} of {len(time)} time bins.")

        cnt_unit = f"counts/{bintime * 1e3:.0f} ms"
        base_meta = {
            "bintime": bintime, "n_exp": n_exp, "lt_bins": bins, "sigma": sigma,
            "lifetime_bins": int(lifetime_bins),
            "intensity_bins": int(intensity_bins),
            "roi": bounds, "time": time, "taus": taus, "counts": counts,
        }

        def _make_image(tau_vals: np.ndarray, label: str) -> Image:
            valid = np.isfinite(tau_vals) & np.isfinite(counts)
            hist, tau_edges, cnt_edges = np.histogram2d(
                tau_vals[valid], counts[valid],
                bins=(int(lifetime_bins), int(intensity_bins)))
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
        self._require_pulsed()
        if not self._tt_cache:
            self.time_trace()
        trace = self._tt_cache["trace"]
        matrix = self._tt_cache["matrix"]
        max_counts = np.max(trace)
        if off == 0 and on == 1:
            on_mask = np.ones(trace.shape, dtype=bool)
            off_mask = np.zeros(trace.shape, dtype=bool)
        else:
            on_mask = trace >= int(on * max_counts)
            off_mask = trace < int(off * max_counts)
        on_matrix = matrix[on_mask, :]
        off_matrix = matrix[off_mask, :]
        bintime = float(self._tt_cache["bintime"])
        durations = np.asarray(self._tt_cache["bin_durations"], dtype=float)

        def _meta(fraction: float, mask: np.ndarray) -> dict[str, float | int]:
            return {
                "fraction": fraction,
                "time_bin_count": int(np.count_nonzero(mask)),
                "bintime": bintime,
                "acquisition_duration": float(np.sum(durations[mask])),
            }

        return {
            "on": Signal(self.lt_time, on_matrix.sum(axis=0),
                         label=f"on ({on:.0%})", units={"x": "s", "y": "counts"},
                         meta=_meta(on, on_mask)),
            "off": Signal(self.lt_time, off_matrix.sum(axis=0),
                          label=f"off ({off:.0%})", units={"x": "s", "y": "counts"},
                          meta=_meta(off, off_mask)),
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
        self._require_pulsed()
        y = np.asarray(histogram["y"], dtype=float)
        t0 = histogram["x"][np.argmax(y)]
        t_ns = (histogram["x"] - t0) * 1e9

        mask = fit_window_mask(t_ns, y, mode, fraction=fraction,
                               smoothing=smoothing, start=start, end=end)

        t_fit, y_fit = t_ns[mask], y[mask]
        norm = 1.0
        if normalize and np.max(y_fit) > 0:
            norm = np.max(y_fit)
            y_fit = y_fit / norm
        duration = histogram["meta"].get("acquisition_duration")
        if self.dark_counts and duration is None:
            raise ValueError(
                "Histogram acquisition duration is required for dark-count correction. "
                "Build it with lifetime_histogram(matrix, bintime=...)."
            )
        offset = self.dark_counts * float(duration or 0.0) / self.n_lt_bins / norm

        fit = MultiExpFit(t_fit, y_fit, n_exp, offset, **fit_kwargs)
        out = fit.to_dict()
        out.update({"shifted_histogram_x": t_ns, "t_fit": t_fit, "y_fit": y_fit, "left_bound_ns": float(t_fit[0])})
        return out

    # -------------------------------------------------------------- triggers
    def laser_triggers(self, marker_channel: int = 4, *,
                       reference: Signal | None = None) -> TriggerCollection | None:
        """Locate the laser triggers in the marker channel (default).
            With only a single marker on the channel, a warning is emitted and the
            left and right triggers coincide (usable as a single-trigger spec for
            :class:`~kpfm_pl.match.Match`). Falls back to the time-trace edges
            (when ``laser_pulse=False`` was used) if no hardware marker is present.
            For this legacy case only the first and last trigger work.
            When ``reference`` is supplied, indices and times are mapped to its
            time axis. Markers outside the reference are rejected rather than
            clamped to an unrelated endpoint.
    
            Parameters
            ----------
            marker_channel :
                The PicoHarps marker channel where the laser trigger is detected.
            reference :
                Optional binned :class:`Signal`; when given, trigger indices are
                mapped onto its (binned) time axis. Default: None
    
            Returns
            -------
            dict or None
            ``{"left": {"index", "time"}, "right": {"index", "time"},
            "all": [{"index", "time"}, ...]}``. The ``"all"`` list holds
            every detected marker in the same point format as ``left`` and
            ``right``. ``None`` if no trigger is found.
        """
        
        self._require_pulsed()
        if self.marker.get(marker_channel):
            markers = self.marker[marker_channel]
            if len(markers) == 1:
                warnings.warn(
                    f"{self.filepath.stem}: only one laser trigger found on "
                    f"marker channel {marker_channel}; left and right triggers "
                    "coincide.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if reference is None and not self._tt_cache:
                self.time_trace()
            if reference is None:
                reference = self._tt_cache["signal"]
            ref_t = np.asarray(reference["x"], dtype=float)
            all_times = [marker*self.rep_time for marker in markers]
            left_t = markers[0] * self.rep_time
            right_t = markers[-1] * self.rep_time
            left_idx = nearest_index(ref_t, left_t)
            right_idx = nearest_index(ref_t, right_t)
            triggers = trigger_dict(left_idx, left_t, right_idx, right_t)
            all_idx = np.unique(np.array([nearest_index(ref_t, t) for t in all_times], dtype=int))
            points: list[TriggerPoint] = [
                        {"index": int(index), "time": float(trigger_time)}
                        for index, trigger_time in zip(all_idx, all_times, strict=True)]
            return {**triggers, "all": points}
        elif self._tt_cache.get("triggers"): # this is the legacy bit. Only works for 2 triggers
            triggers = self._tt_cache["triggers"]
            if reference is not None:
                ref_t = np.asarray(reference["x"], dtype=float)
                left_t = triggers["left"]["time"]
                right_t = triggers["right"]["time"]
                left_idx = nearest_index(ref_t, left_t)
                right_idx = nearest_index(ref_t, right_t)
                triggers = trigger_dict(left_idx, left_t, right_idx, right_t)
        else:
            warnings.warn(
                f"{self.filepath.stem}: no laser triggers found.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        if triggers is None:
            return None
        print(f"{self.filepath.stem}: trigger spacing "
              f"{triggers['right']['time'] - triggers['left']['time']:.2f} s")
        return triggers

    def get_markers(self, marker_channel: int = 4) -> np.ndarray | None:
        """All marker times (seconds) on ``marker_channel``."""
        self._require_pulsed()
        if marker_channel in self.marker:
            return np.array(self.marker[marker_channel]) * self.rep_time
        warnings.warn(
            f"{self.filepath.stem}: marker channel {marker_channel} not found.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
