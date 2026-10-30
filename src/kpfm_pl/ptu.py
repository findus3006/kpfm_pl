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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .core import Image, Signal
from .fitting import MultiExpFit, poisson_chi_squared
from .utils import TriggerPoint, fit_window_mask, lifetime_from_truncated_mean, trigger_points

__all__ = ["PTU", "flid_map"]

_WRAPAROUND = 65536
#: Where ``set_lifetime_shift("auto")`` puts the histogram peak (ns after zero).
AUTO_SHIFT_MARGIN_NS = 5.0

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


def _poisson_weights(weighting: str, counts: np.ndarray, scale: float,
                     fit_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Add Poisson ``weights`` for :class:`MultiExpFit` when ``weighting="poisson"``.

    ``counts`` are the raw photon counts of the fitted bins and ``scale`` the
    factor the fitted data was divided by (1 without normalisation). The
    weighted residuals are then in units of the Poisson standard deviation,
    which also gives a robust ``loss``'s ``f_scale`` a fixed meaning.
    """
    if weighting == "none":
        return fit_kwargs
    if weighting != "poisson":
        raise ValueError(f"weighting must be 'none' or 'poisson', not {weighting!r}.")
    if "weights" in fit_kwargs:
        raise ValueError("Pass either weighting='poisson' or explicit weights, not both.")
    weights = scale / np.sqrt(np.maximum(np.asarray(counts, dtype=float), 1.0))
    return {**fit_kwargs, "weights": weights}


def _weighted_mean_lifetime(taus: np.ndarray, amps: np.ndarray) -> np.ndarray:
    """Intensity-weighted mean lifetime ``sum(A tau^2) / sum(A tau)`` per row.

    Rows with a failed fit (NaN) or zero total weight (all ``A = 0``) give NaN.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.sum(amps * taus**2, axis=1) / np.sum(amps * taus, axis=1)
    mean[~np.isfinite(mean)] = np.nan
    return mean


#: Extra attempts of ``auto_bintime`` after the first suggestion.
_AUTO_BINTIME_STEPS = 3


def _nice_ceil(value: float) -> float:
    """Smallest number of the form 1, 2, 2.5 or 5 x 10^n that is >= ``value``."""
    base = 10.0 ** np.floor(np.log10(value))
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        if step * base >= value * (1 - 1e-9):
            return float(f"{step * base:.12g}")
    return float(value)  # not reached


def _count_check(counts: np.ndarray, bintime: float, min_counts: float,
                 coverage: float) -> dict[str, Any]:
    """Do at least ``coverage`` of the time bins hold ``min_counts`` photons?

    ``level`` is the photon count reached by that fraction of the bins (the
    ``1 - coverage`` quantile). Counts per bin grow linearly with the bin
    time, so ``bintime * min_counts / level`` (rounded up to a round number)
    is the suggested bin time.
    """
    counts = np.asarray(counts, dtype=float)
    level = float(np.quantile(counts, 1.0 - coverage)) if counts.size else 0.0
    passed = bool(level >= min_counts)
    return {
        "min_counts": float(min_counts),
        "coverage": float(coverage),
        "bintime": float(bintime),
        "level": level,
        "fraction": float(np.mean(counts >= min_counts)) if counts.size else 0.0,
        "passed": passed,
        "suggested_bintime": (None if passed
                              else _nice_ceil(bintime * min_counts / max(level, 1.0))),
    }


def _validate_map_args(lifetime_bins, intensity_bins, lifetime_range, intensity_range) -> None:
    for name, value in (("lifetime_bins", lifetime_bins),
                        ("intensity_bins", intensity_bins)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not 1 <= value <= 1024):
            raise ValueError(f"{name} must be an integer between 1 and 1024.")
    for name, value in (("lifetime_range", lifetime_range),
                        ("intensity_range", intensity_range)):
        if value is not None:
            lo_hi = np.asarray(value, dtype=float)
            if lo_hi.shape != (2,) or not np.all(np.isfinite(lo_hi)) or lo_hi[0] >= lo_hi[1]:
                raise ValueError(f"{name} must be a finite (min, max) pair with min < max.")


def flid_map(trace: Signal, *, lifetime_bins: int = 100, intensity_bins: int = 100,
             lifetime_range: tuple[float, float] | None = None,
             intensity_range: tuple[float, float] | None = None,
             sigma: float = 1.5, isolated: bool = False,
             label: str | None = None) -> Image | list[Image]:
    """Build a FLID map from a per-bin lifetime trace, without fitting.

    The second half of :meth:`PTU.flid`: ``trace`` is the :class:`Signal` of
    :meth:`PTU.lifetime_trace` or :meth:`PTU.arrival_time_trace`
    (``meta["counts"]`` holds the photons per bin). Use it to redraw a map
    with other histogram settings, or to compare fitted and mean-arrival
    maps on identical bins.

    Parameters
    ----------
    trace :
        Per-bin lifetimes (``y``, ns) with ``meta["counts"]``; bins with a
        ``NaN`` lifetime are left out.
    lifetime_bins, intensity_bins :
        Number of histogram bins on the lifetime and raw-count axes. Each
        must be between 1 and 1024, which keeps map allocation bounded.
    lifetime_range, intensity_range :
        ``(min, max)`` of the histogram on the lifetime (ns) and raw-count
        axes; values outside are dropped. Pass the same ranges to several
        FLIDs to get identical bins, so their maps can be compared directly.
        ``None`` spans the data, padded by ``ceil(3 * sigma)`` bins on each
        side (not below 0) so the smoothed histogram is not cut off at the
        outermost points.
    sigma :
        Gaussian smoothing applied to the 2-D histogram (``0`` disables).
    isolated :
        One map per fitted component (``meta["taus"]``, fitted traces only)
        instead of one map of ``y``.
    label :
        Map label; default derived from ``trace.label``.

    Returns
    -------
    Image or list[Image]
        As :meth:`PTU.flid`. ``meta`` copies ``trace.meta`` and adds
        ``time``, ``bin_lifetimes`` (the mapped value of every bin), ``sigma``,
        ``lifetime_bins``, ``intensity_bins`` and the ``lifetime_range``/
        ``intensity_range`` actually used (the exact ``imshow`` extent).
    """
    from scipy import ndimage

    _validate_map_args(lifetime_bins, intensity_bins, lifetime_range, intensity_range)
    counts = trace.meta.get("counts")
    if counts is None or len(counts) != len(trace):
        raise ValueError("trace.meta['counts'] must hold the photons of every time bin.")
    counts = np.asarray(counts, dtype=float)
    bintime = trace.meta.get("bintime")
    cnt_unit = f"counts/{bintime * 1e3:g} ms" if bintime else "counts"
    if label is None:
        label = trace.label.rsplit(" mean ", 1)[0] + " FLID"
        if trace.meta.get("method") == "mean_arrival":
            label += " (mean arrival)"
    base_meta = {**trace.meta, "time": np.asarray(trace.x, dtype=float), "sigma": sigma,
                 "lifetime_bins": int(lifetime_bins),
                 "intensity_bins": int(intensity_bins)}

    def _hist_range(values: np.ndarray, given, n_bins: int) -> tuple[float, float]:
        if given is not None:
            return float(given[0]), float(given[1])
        if values.size == 0:
            return 0.0, 1.0
        lo, hi = float(values.min()), float(values.max())
        pad = int(np.ceil(3 * sigma)) if sigma > 0 else 0
        if pad == 0:
            return lo, hi
        # Add pad empty bins on each side so the Gaussian smoothing tails
        # stay inside the map. The maximum sits on the lower edge of its
        # bin, so the data covers n_bins - 2 * pad bins with n_bins - 2 * pad - 1
        # widths between lo and hi, and the range spans n_bins widths.
        width = (hi - lo) / max(n_bins - 2 * pad - 1, 1)
        new_lo = lo - pad * width
        if lo >= 0:
            new_lo = max(new_lo, 0.0)  # lifetimes and counts are never negative
        return new_lo, hi + (pad + 1) * width

    def _make_image(tau_vals: np.ndarray, name: str) -> Image:
        valid = np.isfinite(tau_vals) & np.isfinite(counts)
        tau_range = _hist_range(tau_vals[valid], lifetime_range, int(lifetime_bins))
        cnt_range = _hist_range(counts[valid], intensity_range, int(intensity_bins))
        hist, tau_edges, cnt_edges = np.histogram2d(
            tau_vals[valid], counts[valid],
            bins=(int(lifetime_bins), int(intensity_bins)),
            range=(tau_range, cnt_range))
        if sigma > 0:
            hist = ndimage.gaussian_filter(hist, sigma=sigma)
        tau_centers = 0.5 * (tau_edges[:-1] + tau_edges[1:])
        cnt_centers = 0.5 * (cnt_edges[:-1] + cnt_edges[1:])
        # hist is (n_tau, n_counts); Image data is (ny, nx) = (counts, tau).
        return Image(hist.T, x=tau_centers, y=cnt_centers, label=name,
                     units={"x": "ns", "y": cnt_unit, "z": "occurrence"},
                     meta={**base_meta, "bin_lifetimes": tau_vals,
                           "lifetime_range": (float(tau_edges[0]), float(tau_edges[-1])),
                           "intensity_range": (float(cnt_edges[0]), float(cnt_edges[-1]))})

    if isolated:
        taus = trace.meta.get("taus")
        if taus is None:
            raise ValueError("isolated maps need fitted components (meta['taus']).")
        taus = np.asarray(taus, dtype=float)
        return [_make_image(taus[:, i], f"{label} (tau {i + 1})")
                for i in range(taus.shape[1])]
    return _make_image(np.asarray(trace.y, dtype=float), label)


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
    lifetime_shift :
        Optional circular shift of the lifetime axis, see
        :meth:`set_lifetime_shift` (``None``, the default, keeps the raw data).
    """

    #: Requested and applied lifetime shift, see :meth:`set_lifetime_shift`.
    lifetime_shift: float | str | None = None
    lifetime_shift_ns: float = 0.0
    #: Lifetime bin holding the partial last bin of the period after a shift.
    lifetime_seam_bin: int | None = None

    def __init__(self, filepath: str | Path, dark_counts: float = 0.0,
                 smooth_rep_rate: bool = False,
                 lifetime_shift: float | str | None = None) -> None:
        self.filepath = Path(filepath)
        self.dark_counts = float(dark_counts)
        self._tags: dict[str, Any] = {}
        self.marker: dict[int, list[int]] = {}
        self.events = np.empty((0, 3))
        self._raw_dtime = np.empty(0)
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
        if lifetime_shift is not None:
            self.set_lifetime_shift(lifetime_shift)

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
        self._raw_dtime = self.events[:, 1].copy()
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

    def set_lifetime_shift(self, shift: float | str | None) -> float:
        """Circularly shift every photon's lifetime (micro) time to earlier times.

        A salvage tool for recordings whose sync offset put the decay late in
        the lifetime window (e.g. the peak at 80 ns of a 100 ns period): the
        tail then continues at the start of the window, after the next sync.
        Because the window spans one laser period, rotating it by ``shift``
        restores the full decay. Prefer fixing the sync offset of the
        acquisition.

        Parameters
        ----------
        shift :
            ``None`` (or 0) restores the raw data, a number shifts by that many
            ns (rounded to whole lifetime bins), ``"auto"`` puts the peak of the
            whole-file histogram :data:`AUTO_SHIFT_MARGIN_NS` after zero.

        Returns
        -------
        float
            The applied shift in ns (also :attr:`lifetime_shift_ns`).

        Notes
        -----
        Time traces are unaffected; photons moved across the window edge
        belong to the previous laser pulse, i.e. their arrival time is off by
        one period, which is negligible for time bins of milliseconds.
        If the period is not a whole number of bins, the last bin of the window
        is only partly covered. After a shift it lies inside the decay
        (:attr:`lifetime_seam_bin`), so the lifetime fits leave it out.
        All cached traces are discarded.
        """
        self._require_pulsed()
        period_bins = self.rep_time / self.resolution
        wrap = int(np.ceil(period_bins - 1e-6))
        if wrap > self.n_lt_bins:
            raise ValueError(
                f"{self.filepath.stem}: the lifetime window ({self.n_lt_bins} bins) is "
                f"shorter than one laser period ({period_bins:.0f} bins), so the data is "
                "not periodic and cannot be shifted circularly."
            )
        raw = self._raw_dtime
        if isinstance(shift, str):
            if shift != "auto":
                raise ValueError(f"shift must be a number, None or 'auto', not {shift!r}.")
            hist = np.bincount(raw.astype(int) % wrap, minlength=wrap).astype(float)
            # A short circular box filter keeps single noisy bins from winning.
            padded = np.concatenate((hist[-2:], hist, hist[:2]))
            peak = int(np.argmax(np.convolve(padded, np.ones(5) / 5, mode="valid")))
            margin = int(round(AUTO_SHIFT_MARGIN_NS * 1e-9 / self.resolution))
            bins = (peak - margin) % wrap
        elif shift is None:
            bins = 0
        else:
            if not np.isfinite(float(shift)):
                raise ValueError("shift must be finite.")
            bins = int(round(float(shift) * 1e-9 / self.resolution)) % wrap

        self.events[:, 1] = (raw - bins) % wrap if bins else raw
        self.lifetime_shift = shift if bins else None
        self.lifetime_shift_ns = float(bins * self.resolution * 1e9)
        partial = period_bins - (wrap - 1) < 0.99
        self.lifetime_seam_bin = (wrap - 1 - bins) % wrap if bins and partial else None
        # Matrices and traces were built from the old lifetime times.
        self._bintime = None
        self._tt_cache = {}
        if bins:
            print(f"{self.filepath.stem}: lifetime axis shifted by "
                  f"{self.lifetime_shift_ns:.3f} ns.")
        return self.lifetime_shift_ns

    def _drop_seam(self, mask: np.ndarray, bins: int = 1) -> np.ndarray:
        """Remove the partial seam bin (see :meth:`set_lifetime_shift`) from ``mask``.

        ``mask`` is over lifetime bins grouped by ``bins``.
        """
        seam = self.lifetime_seam_bin
        if seam is None or seam // bins >= mask.size:
            return mask
        mask = mask.copy()
        mask[seam // bins] = False
        return mask

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
            and its rising/falling edges are stored as two triggers in
            ``meta["triggers"]``. This is bad practice and only kept for
            backwards compatibility.
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
                meta["triggers"] = trigger_points(time[[nz2[0], nz2[-1]]], time)
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
        if self.lifetime_shift_ns:
            meta["lifetime_shift_ns"] = self.lifetime_shift_ns
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

    def _bins_in_window(self, bintime: float, xmn: float, xmx: float, ymn: float,
                        ymx: float, laser_pulse: bool) -> dict[str, np.ndarray]:
        """Time bins of ``[xmn, xmx)`` and those with counts in ``[ymn, ymx)``."""
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

        selected = (counts >= ymn) & (counts < ymx)
        if not np.any(selected) or not np.any(matrix[selected, :]):
            raise ValueError("The selected lifetime region contains no photons.")
        return {"time": time, "counts": counts, "matrix": matrix, "selected": selected}

    def _select_time_bins(
        self, roi: dict[str, float] | None, *,
        x_min: float | None, x_max: float | None,
        y_min: float | None, y_max: float | None,
        bintime: float, laser_pulse: bool | None,
        min_counts: float | None, coverage: float, auto_bintime: bool,
    ) -> dict[str, Any]:
        """Time bins of a per-bin lifetime analysis, with the photon-count check.

        Returns a dict with ``time``, ``counts``, ``matrix``, ``selected`` (the
        bins inside the intensity range), ``bounds`` (the ROI actually used),
        ``bintime`` (the bin time actually used), ``laser_pulse`` and
        ``count_check`` (``None`` when ``min_counts`` disables the check).
        """
        self._require_pulsed()
        if roi is not None:
            x_min, x_max = roi.get("x_min"), roi.get("x_max")
            y_min, y_max = roi.get("y_min"), roi.get("y_max")
        if min_counts is not None and (not np.isfinite(min_counts) or min_counts < 0):
            raise ValueError("min_counts must be a non-negative number or None.")
        if not 0 < coverage <= 1:
            raise ValueError("coverage must lie in (0, 1].")

        xmn = 0.0 if x_min is None else x_min
        xmx = self.acquisition_time if x_max is None else x_max
        # Lifetime values must be computed from the same time-bin layout as
        # the PL trace they accompany.  In particular, ``laser_pulse=False``
        # centres and pads legacy marker-less traces.  Rebuilding with the
        # normal default would silently put the lifetime values back on the
        # unshifted bins.  Unless explicitly overridden, inherit the mode of
        # the most recently generated time trace.
        if laser_pulse is None:
            laser_pulse = bool(self._tt_cache.get("laser_pulse", True))
        elif not isinstance(laser_pulse, (bool, np.bool_)):
            raise TypeError("laser_pulse must be a bool or None.")

        requested = float(bintime)
        current = requested
        check = None
        too_short = False
        for attempt in range(_AUTO_BINTIME_STEPS + 1):
            # A count range drawn at the requested bin time keeps its meaning
            # at another bin time when it is scaled like the counts per bin.
            factor = current / requested
            ymn = -np.inf if y_min is None else y_min * factor
            ymx = np.inf if y_max is None else y_max * factor
            sel = self._bins_in_window(current, xmn, xmx, ymn, ymx, bool(laser_pulse))
            if not min_counts:
                break
            check = _count_check(sel["counts"][sel["selected"]], current,
                                 float(min_counts), float(coverage))
            if check["passed"] or not auto_bintime or attempt == _AUTO_BINTIME_STEPS:
                break
            if check["suggested_bintime"] > (xmx - xmn) / 2:
                too_short = True  # fewer than two bins would be left
                break
            current = check["suggested_bintime"]

        stem = self.filepath.stem
        if check is not None:
            check["requested_bintime"] = requested
            if not check["passed"]:
                advice = (f"use a bin time of at least {check['suggested_bintime']:g} s"
                          if not too_short else
                          f"the time window is too short for the {check['suggested_bintime']:g} s "
                          "bin time this would need")
                warnings.warn(
                    f"{stem}: only {check['fraction']:.0%} of the "
                    f"{int(np.count_nonzero(sel['selected']))} time bins of {current:g} s "
                    f"reach {check['min_counts']:g} photons ({check['coverage']:.0%} "
                    f"required); {advice}.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        if current != requested:
            print(f"{stem}: automatic bin time {current:g} s instead of {requested:g} s "
                  f"(count range scaled by {current / requested:g}).")
        factor = current / requested
        bounds = {"x_min": x_min, "x_max": x_max,
                  "y_min": None if y_min is None else y_min * factor,
                  "y_max": None if y_max is None else y_max * factor}
        return {**sel, "bounds": bounds, "bintime": current,
                "laser_pulse": bool(laser_pulse), "count_check": check}

    def _lifetime_window(self, matrix: np.ndarray, mode: str, fraction: float,
                         smoothing: float, start: float, end: float | None,
                         ) -> tuple[np.ndarray, float, np.ndarray]:
        """``(lt_ns, t0_ns, mask)``: the common fit window of the bins in ``matrix``.

        ``mask`` is over the native lifetime bins; the window is chosen on the
        summed histogram of all selected bins, with its maximum at ``t0_ns``.
        """
        lt_ns = self.lt_time * 1e9
        lt_roi = matrix.sum(axis=0)
        t0_ns = float(lt_ns[int(np.argmax(lt_roi))])
        mask = fit_window_mask(lt_ns - t0_ns, lt_roi, mode, fraction=fraction,
                               smoothing=smoothing, start=start, end=end)
        return lt_ns, t0_ns, mask

    def _fit_time_bin_lifetimes(
        self, sel: dict[str, Any], *, n_exp: int = 1, bins: int = 1,
        mode: str = "fraction", fraction: float = 0.95,
        smoothing: float = 2.0, start: float = 0.0,
        end: float | None = None, weighting: str = "none",
        progress: Callable[[int, int], Any] | None = None,
        **fit_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit lifetimes per selected time bin; ``(taus, amps)`` keep all bin positions."""
        time, matrix, selected = sel["time"], sel["matrix"], sel["selected"]
        bintime = sel["bintime"]

        # Use the selected region to choose one common lifetime fit window, then
        # fit each selected row independently. Unselected/invalid/failed rows
        # remain NaN so their positions stay aligned with ``time``.
        lt_ns, t0_ns, mask = self._lifetime_window(
            matrix[selected, :], mode, fraction, smoothing, start, end)

        group_starts = np.arange(0, matrix.shape[1], bins)
        if group_starts.size < 2:
            raise ValueError("bins must leave at least two lifetime groups for fitting.")
        group_sizes = np.minimum(bins, matrix.shape[1] - group_starts)
        binned_lt = np.add.reduceat(lt_ns, group_starts) / group_sizes
        mask = self._drop_seam(np.add.reduceat(mask, group_starts) > 0, bins)
        t_fit = (binned_lt - t0_ns)[mask]
        if t_fit.size < 2:
            raise ValueError("The selected lifetime fit window contains fewer than two bins.")
        binned_matrix = np.add.reduceat(matrix, group_starts, axis=1)

        taus = np.full((len(time), n_exp), np.nan, dtype=float)
        amps = np.full((len(time), n_exp), np.nan, dtype=float)
        n_binned = len(mask)
        rows = np.flatnonzero(selected)
        for done, index in enumerate(rows):
            # Report before each fit so a raising callback aborts promptly.
            if progress is not None:
                progress(done, rows.size)
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
            bin_kwargs = _poisson_weights(weighting, lt_bin[mask], norm * ymax, fit_kwargs)
            fit = MultiExpFit(t_fit, y_fit, n_exp, offset, **bin_kwargs)
            if fit.success:
                params = fit.parameters()
                fitted_taus = np.asarray(params["tau"], dtype=float)
                fitted_amps = np.asarray(params["A"], dtype=float)
                if (fitted_taus.shape == fitted_amps.shape == (n_exp,)
                        and np.all(np.isfinite(fitted_taus))
                        and np.all(np.isfinite(fitted_amps))):
                    taus[index] = fitted_taus
                    amps[index] = fitted_amps
        if progress is not None:
            progress(rows.size, rows.size)
        return taus, amps

    def _validate_fit_args(self, n_exp: int, bins: int, weighting: str,
                           fit_kwargs: dict[str, Any]) -> None:
        # Validate before the (slow) loop rather than at the first fit.
        _poisson_weights(weighting, np.ones(1), 1.0, fit_kwargs)
        if (isinstance(n_exp, (bool, np.bool_))
                or not isinstance(n_exp, (int, np.integer)) or n_exp < 1):
            raise ValueError("n_exp must be a positive integer.")
        if (isinstance(bins, (bool, np.bool_))
                or not isinstance(bins, (int, np.integer)) or bins < 1):
            raise ValueError("bins must be a positive integer.")

    def _trace_meta(self, sel: dict[str, Any], method: str) -> dict[str, Any]:
        return {"bintime": sel["bintime"], "method": method, "roi": sel["bounds"],
                "counts": sel["counts"], "laser_pulse": sel["laser_pulse"],
                "lifetime_shift_ns": self.lifetime_shift_ns,
                "count_check": sel["count_check"]}

    def lifetime_trace(self, roi: dict[str, float] | None = None, *,
                       x_min: float | None = None, x_max: float | None = None,
                       y_min: float | None = None, y_max: float | None = None,
                       bintime: float = 50e-3, n_exp: int = 1, bins: int = 1,
                       mode: str = "fraction", fraction: float = 0.95,
                       smoothing: float = 2.0, start: float = 0.0,
                       end: float | None = None,
                       laser_pulse: bool | None = None,
                       weighting: str = "none",
                       min_counts: float | None = 100, coverage: float = 0.9,
                       auto_bintime: bool = False,
                       progress: Callable[[int, int], Any] | None = None,
                       **fit_kwargs) -> Signal:
        """Return mean fitted lifetime versus acquisition time.

        The returned ``Signal.x`` contains the time-bin positions and
        ``Signal.y`` contains the intensity-weighted mean
        ``sum(A_i * tau_i**2) / sum(A_i * tau_i)`` of the fitted exponential
        lifetimes in ns (equal to ``tau`` for ``n_exp=1``). There is exactly
        one value per time bin. A value is ``NaN`` when the bin is outside the
        requested intensity ROI, contains no usable photons, its fit fails, or
        all its amplitudes are zero. Individual fitted components are
        available as ``signal.meta["taus"]`` and ``signal.meta["amplitudes"]``
        with shape ``(time bins, n_exp)``.
        By default, the trace inherits the ``laser_pulse`` mode of the most
        recent :meth:`time_trace` call.  Pass the option explicitly when no
        time trace has been generated first.

        ``weighting`` (``"none"`` or ``"poisson"``) is applied to every
        per-bin fit as in :meth:`fit_lifetime`. ``progress`` is an optional
        callback ``progress(done, total)`` called before each per-bin fit and
        once at the end; raise an exception from it to abort the computation
        (see :meth:`flid`). ``min_counts``, ``coverage`` and ``auto_bintime``
        check the photons per time bin, see :meth:`flid`.
        """
        self._validate_fit_args(n_exp, bins, weighting, fit_kwargs)
        sel = self._select_time_bins(
            roi, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            bintime=bintime, laser_pulse=laser_pulse, min_counts=min_counts,
            coverage=coverage, auto_bintime=auto_bintime,
        )
        taus, amps = self._fit_time_bin_lifetimes(
            sel, n_exp=n_exp, bins=bins, mode=mode, fraction=fraction,
            smoothing=smoothing, start=start, end=end, weighting=weighting,
            progress=progress, **fit_kwargs,
        )
        return Signal(
            sel["time"], _weighted_mean_lifetime(taus, amps),
            label=f"{self.filepath.stem} mean lifetime",
            units={"x": "s", "y": "ns"},
            meta={**self._trace_meta(sel, "fit"), "n_exp": n_exp, "lt_bins": bins,
                  "taus": taus, "amplitudes": amps},
        )

    def arrival_time_trace(self, roi: dict[str, float] | None = None, *,
                           x_min: float | None = None, x_max: float | None = None,
                           y_min: float | None = None, y_max: float | None = None,
                           bintime: float = 50e-3,
                           mode: str = "fraction", fraction: float = 0.95,
                           smoothing: float = 2.0, start: float = 0.0,
                           end: float | None = None,
                           laser_pulse: bool | None = None,
                           truncation_correction: bool = True,
                           min_counts: float | None = 100, coverage: float = 0.9,
                           auto_bintime: bool = False,
                           progress: Callable[[int, int], Any] | None = None) -> Signal:
        """Lifetime per time bin from the mean photon arrival time, without fitting.

        A fast sanity check of :meth:`lifetime_trace`: same arguments, same
        time bins and the same fit window (chosen on the summed histogram of
        the selected bins). For each bin, the expected dark counts per lifetime
        channel (``dark_counts * bintime / n_lt_bins``) are subtracted and the
        mean delay of the remaining photons after the window start is taken.

        For a decay ``sum(A_i exp(-t/tau_i))`` observed without limit this mean
        is exactly the intensity-weighted mean lifetime
        ``sum(A_i tau_i**2) / sum(A_i tau_i)`` of the fits, independent of where
        the window starts. The window of length ``T`` cuts off late photons, so
        the mean is too short: ``tau - T / (exp(T / tau) - 1)`` for one
        exponential. ``truncation_correction=True`` (default) inverts this
        relation (:func:`~kpfm_pl.utils.lifetime_from_truncated_mean`); it is
        exact for one exponential and approximate for several. Without dark-count
        subtraction (``dark_counts`` set wrongly) the flat background pulls the
        mean towards ``T / 2``.

        Returns
        -------
        Signal
            ``x`` the time-bin positions, ``y`` the lifetime in ns (``NaN`` for
            bins outside the intensity range, without photons, or whose mean
            is not below ``T / 2``). ``meta`` holds ``mean_arrival`` (the
            uncorrected mean in ns), ``photons`` (dark-corrected photons in
            the window), ``window_ns`` (``T``), ``window_start_ns`` (relative
            to the histogram maximum), ``dark_per_channel`` and, as in
            :meth:`lifetime_trace`, ``counts``, ``roi``, ``bintime`` and
            ``count_check``.
        """
        sel = self._select_time_bins(
            roi, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            bintime=bintime, laser_pulse=laser_pulse, min_counts=min_counts,
            coverage=coverage, auto_bintime=auto_bintime,
        )
        time, matrix, selected = sel["time"], sel["matrix"], sel["selected"]
        lt_ns, t0_ns, mask = self._lifetime_window(
            matrix[selected, :], mode, fraction, smoothing, start, end)
        channels = np.flatnonzero(self._drop_seam(mask))
        if channels.size < 2:
            raise ValueError("The selected lifetime fit window contains fewer than two bins.")
        res_ns = self.resolution * 1e9
        window_start = float(lt_ns[channels[0]])
        # Delay of each channel's centre after the window start.
        delay = lt_ns[channels] + 0.5 * res_ns - window_start
        window_ns = float(lt_ns[channels[-1]] + res_ns - window_start)
        dark = self.dark_counts * sel["bintime"] / self.n_lt_bins

        mean = np.full(len(time), np.nan, dtype=float)
        photons = np.full(len(time), np.nan, dtype=float)
        rows = np.flatnonzero(selected)
        chunk = 2048  # bounds the temporary float copy of the matrix
        for begin in range(0, rows.size, chunk):
            if progress is not None:
                progress(begin, rows.size)
            index = rows[begin:begin + chunk]
            block = matrix[np.ix_(index, channels)].astype(float) - dark
            total = block.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                values = (block @ delay) / total
            values[~(total > 0)] = np.nan
            mean[index] = values
            photons[index] = total
        if progress is not None:
            progress(rows.size, rows.size)

        lifetime = (lifetime_from_truncated_mean(mean, window_ns)
                    if truncation_correction else mean.copy())
        return Signal(
            time, lifetime, label=f"{self.filepath.stem} mean arrival time",
            units={"x": "s", "y": "ns"},
            meta={**self._trace_meta(sel, "mean_arrival"), "mean_arrival": mean,
                  "photons": photons, "window_ns": window_ns,
                  "window_start_ns": window_start - t0_ns,
                  "dark_per_channel": dark,
                  "truncation_correction": bool(truncation_correction)},
        )

    def flid(self, roi: dict[str, float] | None = None, *,
             x_min: float | None = None, x_max: float | None = None,
             y_min: float | None = None, y_max: float | None = None,
             bintime: float = 50e-3, n_exp: int = 1, bins: int = 1,
             lifetime_bins: int = 100, intensity_bins: int = 100,
             lifetime_range: tuple[float, float] | None = None,
             intensity_range: tuple[float, float] | None = None,
             mode: str = "fraction", fraction: float = 0.95, smoothing: float = 2.0,
             start: float = 0.0, end: float | None = None, sigma: float = 1.5,
             isolated: bool = False, weighting: str = "none",
             method: str = "fit", truncation_correction: bool = True,
             min_counts: float | None = 100, coverage: float = 0.9,
             auto_bintime: bool = False,
             progress: Callable[[int, int], Any] | None = None,
             **fit_kwargs) -> Image | list[Image]:
        """Compute a FLID (Fluorescence-Lifetime-Intensity-Distribution) map.

        Each time bin of the (optionally) ROI-restricted matrix is fitted with
        ``n_exp`` exponentials; the resulting ``(lifetime, counts)`` pairs are
        binned into a 2-D histogram (lifetime on x, counts on y). Unless
        ``isolated``, the lifetime of a bin is the intensity-weighted mean
        ``sum(A_i * tau_i**2) / sum(A_i * tau_i)`` of its fitted components
        (equal to ``tau`` for ``n_exp=1``).

        This is :meth:`lifetime_trace` (or :meth:`arrival_time_trace`)
        followed by :func:`flid_map`. Call those two yourself to redraw the
        map with other histogram settings without fitting again.

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
        lifetime_bins, intensity_bins, lifetime_range, intensity_range, sigma :
            Histogram settings, see :func:`flid_map`.
        mode, fraction, smoothing, start, end :
            Fit-window left-bound selection, as in :meth:`fit_lifetime`.
        isolated :
            When ``True`` return one :class:`Image` per exponential component;
            otherwise return a single map of the per-bin intensity-weighted
            mean lifetime.
        weighting :
            ``"none"`` (default) or ``"poisson"`` weighting of the per-bin
            fits, as in :meth:`fit_lifetime`.
        method :
            ``"fit"`` (default) fits every bin; ``"mean_arrival"`` uses the
            mean photon arrival time instead (:meth:`arrival_time_trace`, no
            fitting; ``n_exp``, ``bins``, ``weighting`` and ``fit_kwargs`` are
            then ignored and ``isolated`` is not available).
        truncation_correction :
            Only for ``method="mean_arrival"``, see :meth:`arrival_time_trace`.
        min_counts, coverage :
            Photon-count check: at least the fraction ``coverage`` of the
            selected time bins should hold ``min_counts`` photons (``coverage``
            0.5 is the median). Otherwise a ``RuntimeWarning`` suggests the bin
            time that would reach it, assuming counts grow linearly with the
            bin time. ``min_counts=None`` (or 0) disables the check; the result
            is stored in ``meta["count_check"]``.
        auto_bintime :
            When the check fails, recompute with the suggested bin time
            (repeated up to three times, since blinking makes counts grow
            more slowly than the bin time). The ROI's count range is scaled by
            the bin-time ratio, because it was drawn at ``bintime``.
            ``meta["bintime"]`` and ``meta["roi"]`` hold the values used.
        progress :
            Optional callback ``progress(done, total)``, called before each
            per-bin fit and once with ``done == total`` at the end (e.g. to
            drive a progress bar). An exception raised by the callback
            propagates and aborts the computation, which allows cancelling.
        **fit_kwargs :
            Forwarded to :class:`~kpfm_pl.fitting.MultiExpFit`.

        Returns
        -------
        Image or list[Image]
            FLID map(s) with the lifetime axis as ``x`` (ns), counts as ``y``,
            and occurrence as ``data``. Per-bin ``taus``/``amplitudes``/``counts``
            and the histogram ``lifetime_range``/``intensity_range`` actually used are
            stored in ``meta``; ``(*lifetime_range, *intensity_range)`` is the
            exact ``imshow`` extent of the map.
        """
        _validate_map_args(lifetime_bins, intensity_bins, lifetime_range, intensity_range)
        window = {"mode": mode, "fraction": fraction, "smoothing": smoothing,
                  "start": start, "end": end}
        common = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
                  "bintime": bintime, "min_counts": min_counts, "coverage": coverage,
                  "auto_bintime": auto_bintime, "progress": progress, **window}
        if method == "fit":
            trace = self.lifetime_trace(roi, n_exp=n_exp, bins=bins, weighting=weighting,
                                        **common, **fit_kwargs)
            done = "fitted"
        elif method == "mean_arrival":
            if isolated:
                raise ValueError("isolated maps need fitted components (method='fit').")
            trace = self.arrival_time_trace(
                roi, truncation_correction=truncation_correction, **common)
            done = "with a mean arrival time:"
        else:
            raise ValueError(f"method must be 'fit' or 'mean_arrival', not {method!r}.")
        n_ok = int(np.count_nonzero(np.isfinite(trace.y)))
        print(f"{self.filepath.stem}: FLID {done} {n_ok} of {len(trace)} time bins.")
        return flid_map(trace, lifetime_bins=lifetime_bins, intensity_bins=intensity_bins,
                        lifetime_range=lifetime_range, intensity_range=intensity_range,
                        sigma=sigma, isolated=isolated)

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
                     normalize: bool = False, weighting: str = "none",
                     **fit_kwargs) -> dict[str, Any]:
        """Fit a lifetime ``histogram`` with ``n_exp`` exponentials.

        The fit window left bound is chosen by ``mode``: ``"fraction"`` (rising
        edge), ``"inflection"`` or ``"manual"`` (``start``/``end`` in ns). Returns
        the :meth:`~kpfm_pl.fitting.MultiExpFit.to_dict` payload augmented with the
        time/intensity arrays actually fitted.

        ``weighting="poisson"`` weights every histogram bin by its photon
        counting noise ``1 / sqrt(max(counts, 1))`` instead of the default
        unweighted least squares (``"none"``); a robust ``loss``'s
        ``f_scale`` is then in units of that standard deviation. Either way,
        ``out["quality"]`` additionally holds the Poisson goodness of fit from
        :func:`~kpfm_pl.fitting.poisson_chi_squared` (``reduced_chi_squared``,
        ``chi_squared``, ``dof``, ``weighted_residuals`` on ``t_fit``), computed
        on the raw counts even when ``normalize`` is set.
        """
        self._require_pulsed()
        y = np.asarray(histogram["y"], dtype=float)
        t0 = histogram["x"][np.argmax(y)]
        t_ns = (histogram["x"] - t0) * 1e9

        mask = fit_window_mask(t_ns, y, mode, fraction=fraction,
                               smoothing=smoothing, start=start, end=end)
        if y.size == self.n_lt_bins:
            mask = self._drop_seam(mask)

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

        fit_kwargs = _poisson_weights(weighting, y[mask], norm, fit_kwargs)
        fit = MultiExpFit(t_fit, y_fit, n_exp, offset, **fit_kwargs)
        out = fit.to_dict()
        if "curve" in out:
            out.setdefault("quality", {}).update(
                poisson_chi_squared(y[mask], np.asarray(out["curve"]) * norm, 2 * n_exp))
        seam = self.lifetime_seam_bin if y.size == self.n_lt_bins else None
        out.update({"shifted_histogram_x": t_ns, "t_fit": t_fit, "y_fit": y_fit,
                    "left_bound_ns": float(t_fit[0]), "weighting": weighting,
                    "lifetime_shift_ns": self.lifetime_shift_ns,
                    "seam_ns": None if seam is None else float(t_ns[seam])})
        return out

    # -------------------------------------------------------------- triggers
    def laser_triggers(self, marker_channel: int = 4, *,
                       reference: Signal | None = None) -> list[TriggerPoint] | None:
        """Locate the laser triggers in the marker channel (default).

        Every marker on ``marker_channel`` becomes one trigger. If no hardware
        marker is present, falls back to the two time-trace edges stored by
        :meth:`time_trace` with ``laser_pulse=False`` (legacy files).

        Parameters
        ----------
        marker_channel :
            The PicoHarp marker channel where the laser trigger is detected.
        reference :
            Optional binned :class:`Signal`; when given, trigger indices and
            times are mapped onto its time axis. Default: the most recent
            :meth:`time_trace`. Markers outside the reference are dropped with
            a warning rather than clamped to an unrelated endpoint.

        Returns
        -------
        list[TriggerPoint] or None
            Every trigger as ``[{"index", "time"}, ...]`` in chronological
            order (see :func:`~kpfm_pl.utils.trigger_points`). ``None`` if no
            trigger is found. Use :meth:`get_markers` for the exact hardware
            marker times.
        """
        self._require_pulsed()
        if self.marker.get(marker_channel):
            times = np.asarray(self.marker[marker_channel], dtype=float) * self.rep_time
        elif self._tt_cache.get("triggers"):
            # Legacy laser_pulse=False trace: its two edges act as triggers.
            times = np.array([point["time"] for point in self._tt_cache["triggers"]])
        else:
            warnings.warn(
                f"{self.filepath.stem}: no laser triggers found.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        if reference is None:
            if not self._tt_cache:
                self.time_trace()
            reference = self._tt_cache["signal"]
        return trigger_points(times, np.asarray(reference["x"], dtype=float)) or None

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
