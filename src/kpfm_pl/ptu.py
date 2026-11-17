"""Reader for PicoQuant PTU (TTTR T3 and T2) files.

Loads photon records and exposes them as binned time traces, lifetime
histograms and FLID maps. All public methods return :class:`~kpfm_pl.core.Signal`
or :class:`~kpfm_pl.core.Image` objects (or plain dicts) -- there is no plotting
here; feed the returned arrays into matplotlib yourself.

The record format is taken from the header (``TTResultFormat_TTTRRecType``):
T3 and T2 records of the PicoHarp 300 and of the HydraHarp / TimeHarp 260 /
MultiHarp / PicoHarp 330 family. T2 files are converted on load into the same
internal format as T3 files, so everything downstream works alike.
"""

from __future__ import annotations

import struct
import time as _time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .core import Image, Signal
from .fitting import (
    MEAN_LIFETIMES,
    IRFReconvolutionFit,
    MLEFallbackWarning,
    MultiExpFit,
    mean_lifetime,
    poisson_chi_squared,
    prepare_irf,
)
from .utils import (
    TriggerPoint,
    fit_window_mask,
    lifetime_from_truncated_mean,
    photon_correlation,
    prerise_mask,
    pulsed_g2_zero,
    trigger_points,
)

__all__ = ["PTU", "flid_map"]

_WRAPAROUND = 65536
#: Time-tag wraparound of the T2 records (PicoHarp 300; generic format V1, V2).
_T2_WRAPAROUND_PICOHARP = 210698240
_T2_WRAPAROUND_V1 = 33552000
_T2_WRAPAROUND_V2 = 33554432
#: Sync-counter wraparound of the generic (HydraHarp-family) T3 records.
_T3_WRAPAROUND_GENERIC = 1024
#: Records decoded at once (bounds the temporary arrays of large files).
_CHUNK_RECORDS = 1 << 22
#: Most lifetime bins the automatic lifetime binning allows.
MAX_LIFETIME_BINS = 4096
#: Sync intervals used to measure the laser period of a T2 file.
_SYNC_SAMPLES = 1_000_000

#: ``TTResultFormat_TTTRRecType`` -> (measurement mode, record layout, layout
#: version, device). ``"generic"`` is the layout shared by the HydraHarp,
#: TimeHarp 260, MultiHarp and PicoHarp 330.
RECORD_TYPES: dict[int, tuple[str, str, int, str]] = {
    0x00010303: ("T3", "picoharp", 1, "PicoHarp 300"),
    0x00010203: ("T2", "picoharp", 1, "PicoHarp 300"),
    0x00010304: ("T3", "generic", 1, "HydraHarp (V1)"),
    0x00010204: ("T2", "generic", 1, "HydraHarp (V1)"),
    0x01010304: ("T3", "generic", 2, "HydraHarp (V2)"),
    0x01010204: ("T2", "generic", 2, "HydraHarp (V2)"),
    0x00010305: ("T3", "generic", 2, "TimeHarp 260 N"),
    0x00010205: ("T2", "generic", 2, "TimeHarp 260 N"),
    0x00010306: ("T3", "generic", 2, "TimeHarp 260 P"),
    0x00010206: ("T2", "generic", 2, "TimeHarp 260 P"),
    0x00010307: ("T3", "generic", 2, "MultiHarp / PicoHarp 330"),
    0x00010207: ("T2", "generic", 2, "MultiHarp / PicoHarp 330"),
}
#: Files without the record-type tag are read as PicoHarp 300 T3 (as always).
_DEFAULT_RECORD_TYPE = 0x00010303
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


#: Accepted ``weighting`` values of the lifetime fits.
WEIGHTINGS = ("none", "poisson", "mle")
#: ``weighting`` of :meth:`PTU.fit_lifetime` unless another one is passed.
DEFAULT_WEIGHTING = "mle"
#: ``weighting`` of the per-bin fits (:meth:`PTU.lifetime_trace`,
#: :meth:`PTU.flid`) unless another one is passed: as good as the MLE with
#: about 100 photons per bin or more, and several times faster.
DEFAULT_BIN_WEIGHTING = "poisson"


def _resolve_weighting(weighting: str, fit_kwargs: dict[str, Any], name: str = "") -> str:
    """The ``weighting`` a lifetime fit will actually use.

    The Poisson maximum-likelihood fit (``"mle"``, the default) has no
    robust loss and sets its own weights. Combined with a robust ``loss`` it
    therefore falls back to ``"poisson"`` (weighted least squares), and with
    explicit ``weights`` to ``"none"`` (least squares with those weights);
    both are reported as a ``RuntimeWarning``.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, not {weighting!r}.")
    if weighting != "mle":
        return weighting
    prefix = f"{name}: " if name else ""
    loss = fit_kwargs.get("loss", "linear")
    if fit_kwargs.get("weights") is not None:
        warnings.warn(
            f"{prefix}the Poisson maximum-likelihood fit (weighting='mle') sets its own "
            "weights; the explicit weights are used in a least-squares fit instead "
            "(weighting='none').", RuntimeWarning, stacklevel=3)
        return "none"
    if loss != "linear":
        warnings.warn(
            f"{prefix}the Poisson maximum-likelihood fit (weighting='mle') has no robust "
            f"loss; loss={loss!r} is fitted by Poisson-weighted least squares instead "
            "(weighting='poisson').", RuntimeWarning, stacklevel=3)
        return "poisson"
    return "mle"


def _poisson_weights(weighting: str, counts: np.ndarray, scale: float,
                     fit_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Add the :class:`MultiExpFit` arguments of a ``weighting``.

    ``"poisson"`` adds Poisson ``weights``: ``counts`` are the raw photon
    counts of the fitted bins and ``scale`` the factor the fitted data was
    divided by (1 without normalisation). The weighted residuals are then in
    units of the Poisson standard deviation, which also gives a robust
    ``loss``'s ``f_scale`` a fixed meaning. ``"mle"`` selects the Poisson
    maximum-likelihood fit, with ``scale`` converting the data back to counts.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, not {weighting!r}.")
    if weighting == "none":
        return fit_kwargs
    if "weights" in fit_kwargs:
        raise ValueError(f"Pass either weighting={weighting!r} or explicit weights, not both.")
    if weighting == "mle":
        # A robust loss or explicit weights never get here: see _resolve_weighting.
        # The callers report a fallback to least squares themselves (once per
        # call, with the file name), so the fit itself stays quiet.
        return {**fit_kwargs, "statistic": "poisson_mle", "count_scale": float(scale),
                "warn_fallback": False}
    weights = scale / np.sqrt(np.maximum(np.asarray(counts, dtype=float), 1.0))
    return {**fit_kwargs, "weights": weights}


def _weighted_mean_lifetime(taus: np.ndarray, amps: np.ndarray,
                            kind: str = "intensity") -> np.ndarray:
    """Mean lifetime per row (:func:`~kpfm_pl.fitting.mean_lifetime`).

    ``kind="intensity"``: ``sum(A tau^2) / sum(A tau)``; ``"amplitude"``:
    ``sum(A tau) / sum(A)``. Rows with a failed fit (NaN) or zero total
    weight (all ``A = 0``) give NaN.
    """
    return np.asarray(mean_lifetime(amps, taus, kind), dtype=float)


def _check_mean_lifetime(kind: str) -> None:
    if kind not in MEAN_LIFETIMES:
        raise ValueError(f"mean_lifetime must be one of {MEAN_LIFETIMES}, not {kind!r}.")


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


def _decode_records(raw: np.ndarray, mode: str, layout: str, version: int,
                    overflow: int) -> dict[str, Any]:
    """Decode a block of 32-bit TTTR records (vectorised).

    Follows PicoQuant's reference reader ``Read_PTU.py``. ``raw`` holds the
    records as unsigned integers, ``overflow`` the counter offset carried in
    from the records before this block.

    Returns
    -------
    dict
        ``channel`` (detector number of every photon, 1-based), ``counter``
        (its sync number for T3, its time tag for T2, overflow corrected),
        ``dtime`` (T3 only: lifetime bin, else ``None``), ``marker_bits`` and
        ``marker_counter`` (the marker records), ``sync`` (T2, generic layout:
        the time tags of the sync events) and ``overflow`` (the offset to
        carry into the next block).
    """
    raw = np.asarray(raw, dtype=np.uint32)
    dtime = None
    if layout == "picoharp":
        ch = (raw >> 28).astype(np.uint8)
        special = ch == 0xF
        if mode == "T3":
            dtime = ((raw >> 16) & 0xFFF).astype(np.int64)
            counter = (raw & 0xFFFF).astype(np.int64)
            marker_bits = dtime                 # a special record's dtime holds the markers
            wrap = _WRAPAROUND
            channel = ch                        # routing channel, 1-based in the file
        else:
            counter = (raw & 0x0FFFFFFF).astype(np.int64)
            # The lower 4 bits of a special record are the marker bits; they
            # make its time tag wrong by at most 15 ticks, which is ignored.
            marker_bits = counter & 0xF
            wrap = _T2_WRAPAROUND_PICOHARP
            channel = ch + 1                    # the two inputs are 0 and 1 in the file
        is_overflow = special & (marker_bits == 0)
        steps = is_overflow.astype(np.int64)
        is_marker = special & ~is_overflow
        is_sync = np.zeros(raw.shape, dtype=bool)
    elif layout == "generic":
        special = (raw >> 31).astype(bool)
        ch = ((raw >> 25) & 0x3F).astype(np.uint8)
        if mode == "T3":
            dtime = ((raw >> 10) & 0x7FFF).astype(np.int64)
            counter = (raw & 0x3FF).astype(np.int64)
            wrap = _T3_WRAPAROUND_GENERIC
        else:
            counter = (raw & 0x1FFFFFF).astype(np.int64)
            wrap = _T2_WRAPAROUND_V1 if version == 1 else _T2_WRAPAROUND_V2
        is_overflow = special & (ch == 0x3F)
        steps = is_overflow.astype(np.int64)
        if version != 1:
            # V2 stores the number of overflows in the counter field (0 = one).
            steps[is_overflow] = np.maximum(counter[is_overflow], 1)
        is_marker = special & (ch >= 1) & (ch <= 15)
        marker_bits = ch.astype(np.int64)
        # The sync is a regular time-tagged event in T2 (not recorded in T3).
        is_sync = special & (ch == 0) if mode == "T2" else np.zeros(raw.shape, dtype=bool)
        channel = ch + 1                        # input channels are 0-based in the file
    else:
        raise ValueError(f"Unknown record layout {layout!r}.")

    counter = counter + (int(overflow) + np.cumsum(steps)) * wrap
    photon = ~special
    return {
        "channel": channel[photon].astype(np.uint8),
        "counter": counter[photon],
        "dtime": None if dtime is None else dtime[photon],
        "marker_bits": np.asarray(marker_bits)[is_marker].astype(np.int64),
        "marker_counter": counter[is_marker],
        "sync": counter[is_sync],
        "overflow": int(overflow) + int(steps.sum()),
    }


def _auto_binning(units_per_period: float) -> int:
    """Smallest power of two that leaves at most :data:`MAX_LIFETIME_BINS` lifetime bins."""
    binning = 1
    while int(units_per_period / binning) + 1 > MAX_LIFETIME_BINS:
        binning *= 2
    return binning


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
        Gaussian smoothing applied to the 2-D histogram, in map bins (``0``
        disables). It adds its width to the shot-noise spread of the per-bin
        lifetimes (about ``tau / sqrt(N)``), so use ``0`` before judging the
        width of a map.
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
    """Load a PicoQuant ``.ptu`` TTTR file (T3 or T2 mode).

    The header tells how the records are decoded
    (:attr:`measurement_mode`, :attr:`device`, see :data:`RECORD_TYPES`):

    * **T3** -- every photon carries the number of its laser pulse and its
      delay after it (PicoHarp 300; HydraHarp, TimeHarp 260, MultiHarp,
      PicoHarp 330).
    * **T2** -- every event carries an absolute time tag. If the laser pulses
      were recorded (the sync input, channel 0 of the HydraHarp / MultiHarp
      family), the T3 information is recovered on load: the delay of a photon
      is its time since the last laser pulse. The file then behaves like a
      T3 file. Without them (PicoHarp 300: both inputs are detectors; or a
      sync input that is not connected) there is no lifetime information:
      time traces, markers and triggers work, lifetime methods raise a clear
      error (:attr:`has_lifetime` is ``False``).

    Detectors are numbered from 1 (:attr:`channels`); by default the photons
    of all detectors are used together, see :meth:`set_channels`.

    Parameters
    ----------
    filepath :
        Path to the ``.ptu`` file.
    dark_counts :
        Dark counts per second, subtracted as a constant offset during lifetime
        fitting.
    smooth_rep_rate :
        When ``True`` use the rounded laser repetition time for record decoding
        (T3 files only).
    lifetime_shift :
        Optional circular shift of the lifetime axis, see
        :meth:`set_lifetime_shift` (``None``, the default, keeps the raw data).
    channels :
        Detector channel(s) to use, see :meth:`set_channels` (``None``, the
        default: all detectors summed).
    lifetime_binning :
        Number of native time units combined into one lifetime bin (the
        TCSPC resolution of a T3 file, the time-tag unit of a T2 file).
        ``None`` (default) keeps the native bins of a PicoHarp 300 T3 file
        and otherwise takes the smallest power of two that leaves at most
        :data:`MAX_LIFETIME_BINS` bins per laser period.
    sync_divider :
        T2 files with recorded laser pulses: number of laser pulses per
        recorded sync event. ``None`` (default) derives it from the sync
        rate in the header (1 if every pulse was recorded).
    """

    #: Requested and applied lifetime shift, see :meth:`set_lifetime_shift`.
    lifetime_shift: float | str | None = None
    lifetime_shift_ns: float = 0.0
    #: Lifetime bin holding the partial last bin of the period after a shift.
    lifetime_seam_bin: int | None = None
    #: ``"T3"`` or ``"T2"``, from the header.
    measurement_mode: str = "T3"
    #: Device family named by the record type.
    device: str = "PicoHarp 300"
    #: Whether photons carry a lifetime (delay after the laser pulse).
    has_lifetime: bool = True
    #: Detector channels selected with :meth:`set_channels` (``None``: all).
    selected_channels: tuple[int, ...] | None = None
    #: Detector channels that hold photons in this file.
    channels: tuple[int, ...] = ()
    #: Time-tag unit of a T2 file in seconds (``None`` for T3).
    timetag_resolution: float | None = None
    #: Native time units per lifetime bin, see ``lifetime_binning``.
    lifetime_binning: int = 1
    #: Laser pulses per recorded sync event (T2 files with a laser sync).
    sync_divider: int = 1
    #: Sync events recorded in a T2 file.
    sync_events: int = 0
    # Seconds per unit of ``events[:, 0]`` and of the marker counts; ``None``
    # means the laser period ``rep_time`` (T3 files).
    _sync_unit: float | None = None
    _marker_unit: float | None = None
    _all_events: np.ndarray | None = None
    _all_raw_dtime: np.ndarray | None = None
    _all_channel: np.ndarray | None = None

    def __init__(self, filepath: str | Path, dark_counts: float = 0.0,
                 smooth_rep_rate: bool = False,
                 lifetime_shift: float | str | None = None, *,
                 channels: int | Sequence[int] | None = None,
                 lifetime_binning: int | None = None,
                 sync_divider: int | None = None) -> None:
        self.filepath = Path(filepath)
        self.dark_counts = float(dark_counts)
        self._tags: dict[str, Any] = {}
        self.marker: dict[int, list[int]] = {}
        self.events = np.empty((0, 3))
        self._raw_dtime = np.empty(0)
        self._all_events = self.events
        self._all_raw_dtime = self._raw_dtime
        self._all_channel = np.empty(0, dtype=np.uint8)
        self._tt_cache: dict[str, Any] = {}
        for name, value in (("lifetime_binning", lifetime_binning),
                            ("sync_divider", sync_divider)):
            if value is not None and (isinstance(value, (bool, np.bool_))
                                      or not isinstance(value, (int, np.integer))
                                      or value < 1):
                raise ValueError(f"{name} must be a positive integer or None.")

        with open(self.filepath, "rb") as fh:
            self._read_header(fh)
            self.is_cw = bool(self.measurement_mode == "T3"
                              and np.isinf(self.coarse_rep_time))
            if self.is_cw:
                # Continuous-wave file: no sync period, so records cannot be
                # decoded into lifetime bins. Keep safe defaults so the object
                # is fully constructed; binning methods raise a clear error.
                self.n_lt_bins = 0
                self.lt_bins = np.array([], dtype=int)
                self.lt_time = np.array([], dtype=float)
                print(f"{self.filepath.stem}: continuous-wave data, no trigger present.")
                return
            self._read_records(fh, smooth_rep_rate, lifetime_binning, sync_divider)
        if channels is not None:
            self.set_channels(channels)
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
        record_type = int(self._tags.get("TTResultFormat_TTTRRecType",
                                         _DEFAULT_RECORD_TYPE)) & 0xFFFFFFFF
        if record_type not in RECORD_TYPES:
            known = ", ".join(f"0x{key:08X} ({value[3]} {value[0]})"
                              for key, value in RECORD_TYPES.items())
            raise ValueError(
                f"{self.filepath}: unsupported TTTR record type 0x{record_type:08X}. "
                f"Supported: {known}.")
        self.record_type = record_type
        self.measurement_mode, layout, version, self.device = RECORD_TYPES[record_type]
        self._record_layout = (layout, version)
        self.acquisition_time = self._tags["MeasDesc_AcquisitionTime"] * 1e-3
        global_resolution = float(self._tags["MeasDesc_GlobalResolution"])
        if self.measurement_mode == "T2":
            # T2: the global resolution is the unit of the time tags; the
            # laser period is measured from the recorded sync events.
            if not np.isfinite(global_resolution) or global_resolution <= 0:
                raise ValueError(
                    f"{self.filepath}: invalid time-tag resolution {global_resolution!r} "
                    "(MeasDesc_GlobalResolution) of a T2 file.")
            self.timetag_resolution = global_resolution
            self.coarse_rep_time = self.rep_time = float("inf")
            self.resolution = global_resolution
            return
        self.coarse_rep_time = global_resolution
        if np.isinf(self.coarse_rep_time):
            # Continuous-wave file: keep the infinite rep time but still expose
            # the remaining header attributes so the object is usable.
            self.rep_time = self.coarse_rep_time
        else:
            digits = int(-np.floor(np.log10(abs(self.coarse_rep_time))))
            self.rep_time = round(self.coarse_rep_time, digits)
        self.resolution = self._tags["MeasDesc_Resolution"]

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
    def _record_count(self, fh) -> int:
        """Number of records to read, bounded by the bytes left in the file."""
        declared = int(self.num_records)
        if declared < 0:
            raise ValueError(f"{self.filepath}: invalid number of records {declared}.")
        cur = fh.tell()
        end = fh.seek(0, 2)
        fh.seek(cur)
        available = (end - cur) // 4
        if available < declared:
            warnings.warn(
                f"{self.filepath.stem}: file ended earlier than expected "
                f"({self.num_records} records declared in header).",
                RuntimeWarning,
                stacklevel=4,
            )
        return min(declared, available)

    def _read_records(self, fh, smooth_rep_rate: bool, lifetime_binning: int | None = None,
                      sync_divider: int | None = None) -> None:
        mode = self.measurement_mode
        layout, version = self._record_layout
        n_records = self._record_count(fh)

        channels: list[np.ndarray] = []
        counters: list[np.ndarray] = []
        delays: list[np.ndarray] = []           # T3: lifetime bin; T2: ticks since the sync
        marker_bits: list[np.ndarray] = []
        marker_counts: list[np.ndarray] = []
        overflow = 0
        # T2: the sync events are not stored (there can be billions); each
        # block only needs the last sync before it.
        last_sync: int | None = None
        first_sync: int | None = None
        n_sync = 0
        sync_diffs: list[np.ndarray] = []
        n_diffs = 0
        for begin in range(0, n_records, _CHUNK_RECORDS):
            count = min(_CHUNK_RECORDS, n_records - begin)
            raw = np.frombuffer(self._read_exact(fh, 4 * count, "records"), dtype="<u4")
            part = _decode_records(raw, mode, layout, version, overflow)
            overflow = part["overflow"]
            channels.append(part["channel"])
            counters.append(part["counter"])
            marker_bits.append(part["marker_bits"])
            marker_counts.append(part["marker_counter"])
            if mode == "T3":
                delays.append(part["dtime"])
                continue
            sync = np.sort(part["sync"])
            if last_sync is not None:
                sync = np.concatenate(([last_sync], sync))
            # Time since the last laser pulse; -1 marks photons before the first one.
            index = np.searchsorted(sync, part["counter"], side="right") - 1
            delay = np.full(part["counter"].shape, -1, dtype=np.int64)
            found = index >= 0
            if sync.size:
                delay[found] = part["counter"][found] - sync[index[found]]
                if n_diffs < _SYNC_SAMPLES and sync.size > 1:
                    sync_diffs.append(np.diff(sync)[:_SYNC_SAMPLES - n_diffs])
                    n_diffs += sync_diffs[-1].size
                if first_sync is None:
                    first_sync = int(sync[0])
                n_sync += sync.size - (last_sync is not None)
                last_sync = int(sync[-1])
            delays.append(delay)

        def joined(parts: list[np.ndarray], dtype) -> np.ndarray:
            return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)

        channel = joined(channels, np.uint8)
        counter = joined(counters, np.int64)
        delay = joined(delays, np.int64)
        bits = joined(marker_bits, np.int64)
        counts = joined(marker_counts, np.int64)
        for value in dict.fromkeys(bits.tolist()):      # in order of first appearance
            self.marker[int(value)] = counts[bits == value].tolist()

        if mode == "T3":
            note = self._finish_t3(channel, counter, delay, smooth_rep_rate, lifetime_binning)
        else:
            self.sync_events = n_sync
            note = self._finish_t2(
                channel, counter, delay, first_sync=first_sync, last_sync=last_sync,
                sync_diffs=joined(sync_diffs, np.int64), lifetime_binning=lifetime_binning,
                sync_divider=sync_divider)
        self.events = self._all_events
        self._raw_dtime = self._all_raw_dtime
        self.channels = tuple(int(c) for c in np.unique(self._all_channel))
        print(f"{self.filepath.stem}: {len(self.events)} photons, "
              f"{sum(len(v) for v in self.marker.values())} markers.{note}")

    def _set_lifetime_axis(self, n_lt_bins: int) -> None:
        self.n_lt_bins = int(n_lt_bins)
        self.lt_bins = np.arange(self.n_lt_bins)
        self.lt_time = self.lt_bins * self.resolution

    def _channel_note(self) -> str:
        """Photons per detector for the load message (empty for a single channel 1)."""
        found, counts = np.unique(self._all_channel, return_counts=True)
        if found.size == 0 or (found.size == 1 and found[0] == 1):
            return ""
        return " Photons per channel: " + ", ".join(
            f"{int(c)}: {int(n)}" for c, n in zip(found, counts)) + "."

    def _finish_t3(self, channel: np.ndarray, truesync: np.ndarray, dtime: np.ndarray,
                   smooth_rep_rate: bool, lifetime_binning: int | None) -> str:
        """Store decoded T3 photons; returns a note for the load message."""
        layout, _ = self._record_layout
        rep_time = self.rep_time if smooth_rep_rate else self.coarse_rep_time
        self.rep_time = rep_time
        native = self.resolution
        if lifetime_binning is None:
            # PicoHarp 300 files keep their native bins (12-bit delay, as always).
            binning = 1 if layout == "picoharp" else _auto_binning(rep_time / native)
        else:
            binning = int(lifetime_binning)
        self.lifetime_binning = binning
        self.resolution = native * binning
        limit = (0x1000 if layout == "picoharp" else 0x8000)
        self._set_lifetime_axis(min(int(rep_time / self.resolution) + 1, limit))
        if binning > 1:
            dtime = dtime // binning
        keep = dtime < self.n_lt_bins
        truesync, dtime = truesync[keep], dtime[keep]
        self._all_channel = channel[keep]
        self._all_events = np.column_stack(
            (truesync.astype(float), dtime.astype(float), truesync * rep_time))
        self._all_raw_dtime = self._all_events[:, 1].copy()
        note = self._channel_note()
        if layout != "picoharp" or binning > 1:
            note += (f" {self.device} T3, {self.n_lt_bins} lifetime bins of "
                     f"{self.resolution * 1e12:.4g} ps.")
        return note

    def _finish_t2(self, channel: np.ndarray, timetag: np.ndarray, delay: np.ndarray, *,
                   first_sync: int | None, last_sync: int | None, sync_diffs: np.ndarray,
                   lifetime_binning: int | None, sync_divider: int | None) -> str:
        """Convert decoded T2 photons to the T3-like event table.

        With recorded laser pulses the delay of every photon after its pulse
        becomes the lifetime bin and the file behaves like a T3 file.
        Without them ``events[:, 0]`` is the time tag itself and there is a
        single (meaningless) lifetime bin.
        """
        tick = self.timetag_resolution
        self._all_channel = channel
        self._marker_unit = tick
        time_s = timetag * tick
        note = self._channel_note()

        spacing = None                          # recorded sync interval in ticks
        if sync_diffs.size:
            median = float(np.median(sync_diffs))
            regular = median > 0 and np.mean(np.abs(sync_diffs - median) <= 0.01 * median) >= 0.9
            if regular:
                near = np.abs(sync_diffs - median) <= 0.5 * median
                spacing = float(np.mean(sync_diffs[near]))
                # All sync events give a more precise mean, unless some are missing.
                overall = (last_sync - first_sync) / max(self.sync_events - 1, 1)
                if abs(overall / spacing - 1.0) < 1e-6:
                    spacing = float(overall)
            else:
                warnings.warn(
                    f"{self.filepath.stem}: the sync events (channel 0) are not periodic, "
                    "so they are not used as laser pulses; no lifetimes.",
                    RuntimeWarning, stacklevel=4)
        if spacing is None:
            # No laser pulses: time traces and triggers only.
            self.has_lifetime = False
            self._sync_unit = tick
            self.rep_time = self.coarse_rep_time = float("inf")
            self.resolution = tick
            self._set_lifetime_axis(1)
            self._all_events = np.column_stack(
                (timetag.astype(float), np.zeros(timetag.size), time_s))
            self._all_raw_dtime = self._all_events[:, 1].copy()
            return (f"{note} {self.device} T2 without laser sync: time traces only, "
                    "no lifetimes.")

        if sync_divider is None:
            # The header sync rate is that of the laser; a divider shows as a
            # recorded interval of several laser periods.
            rate = self._tags.get("TTResult_SyncRate")
            ratio = spacing * tick * float(rate) if rate else 1.0
            divider = max(int(round(ratio)), 1)
            if divider > 1 and abs(ratio - divider) > 0.05:
                warnings.warn(
                    f"{self.filepath.stem}: the recorded sync interval is {ratio:.3f} laser "
                    "periods of the header sync rate; assuming every pulse was recorded "
                    "(pass sync_divider= to override).", RuntimeWarning, stacklevel=4)
                divider = 1
        else:
            divider = int(sync_divider)
        self.sync_divider = divider
        period = spacing / divider              # laser period in ticks
        self.rep_time = self.coarse_rep_time = period * tick
        binning = (_auto_binning(period) if lifetime_binning is None
                   else int(lifetime_binning))
        self.lifetime_binning = binning
        self.resolution = tick * binning
        self._set_lifetime_axis(int(period / binning) + 1)

        # Photons before the first recorded pulse take their delay from it;
        # by periodicity the (negative) time folds to the right lifetime.
        early = delay < 0
        if early.any():
            delay = delay.copy()
            delay[early] = timetag[early] - first_sync
        micro = np.mod(delay.astype(float), period)
        dtime = np.minimum(np.floor(micro / binning), self.n_lt_bins - 1)
        # The pulse number on an ideal grid: times a period it is the arrival
        # time to within one period, as for a T3 file.
        truesync = np.floor(timetag / period)
        self._all_events = np.column_stack((truesync, dtime, time_s))
        self._all_raw_dtime = self._all_events[:, 1].copy()
        divided = f", sync divider {divider}" if divider > 1 else ""
        return (f"{note} {self.device} T2 with laser sync: period "
                f"{self.rep_time * 1e9:.6g} ns{divided}, {self.n_lt_bins} lifetime bins of "
                f"{self.resolution * 1e12:.4g} ps.")

    # ------------------------------------------------------------------ public
    @property
    def tags(self) -> dict[str, Any]:
        """All header tags as a ``{name: value}`` dict."""
        return dict(self._tags)

    def get_events(self) -> np.ndarray:
        """Photon events of the selected channels, shape ``(N, 3)`` = ``[truesync, dtime, time_s]``.

        ``truesync`` is the number of the laser pulse, ``dtime`` the lifetime
        bin and ``time_s`` the arrival time in seconds. T2 files are converted
        to the same layout: with recorded laser pulses ``dtime`` is the binned
        delay after the last pulse and ``time_s`` the exact time tag; without
        them ``truesync`` is the time tag itself and ``dtime`` is 0.
        """
        return self.events

    def set_channels(self, channels: int | Sequence[int] | None = None) -> tuple[int, ...]:
        """Choose the detector channel(s) whose photons are analysed.

        All time traces, lifetime histograms, fits and FLIDs use the photons
        of the selected detectors together. Markers and laser triggers do not
        depend on the selection, and a lifetime shift is applied again
        (``"auto"`` is determined from the selected photons).

        Parameters
        ----------
        channels :
            One detector number, several, or ``None`` for all detectors of
            the file (:attr:`channels`, numbered from 1).

        Returns
        -------
        tuple of int
            The channels in use. All cached traces are discarded.
        """
        self._require_pulsed()
        if channels is None:
            chosen = None
        else:
            values = np.atleast_1d(np.asarray(channels))
            if values.size == 0 or not np.issubdtype(values.dtype, np.integer):
                raise ValueError("channels must be detector numbers (integers) or None.")
            chosen = tuple(sorted({int(c) for c in values}))
            missing = [c for c in chosen if c not in self.channels]
            if missing:
                raise ValueError(
                    f"{self.filepath.stem}: no photons on channel(s) {missing}; "
                    f"the file has channel(s) {list(self.channels)}.")
        if chosen is None:
            self.events, self._raw_dtime = self._all_events, self._all_raw_dtime
        else:
            keep = np.isin(self._all_channel, chosen)
            self.events, self._raw_dtime = self._all_events[keep], self._all_raw_dtime[keep]
        self.selected_channels = chosen
        # Matrices and traces were built from the old selection.
        self._bintime = None
        self._tt_cache = {}
        if self.has_lifetime:
            # Restores the raw lifetimes of a table shared with the full data
            # and re-applies the requested shift to the selection.
            self.set_lifetime_shift(self.lifetime_shift)
        return self.channels if chosen is None else chosen

    @property
    def photon_channel(self) -> np.ndarray:
        """Detector channel of every photon of the file (all channels, in record order)."""
        return self._all_channel

    def photon_times(self, channel: int | None = None) -> np.ndarray:
        """Absolute arrival time of every photon in seconds, at full resolution.

        The raw material of correlation analyses between detectors (e.g.
        antibunching): ``photon_times(1)`` and ``photon_times(2)`` are the
        photon streams of the two detectors. Independent of
        :meth:`set_channels` and of a lifetime shift.

        Parameters
        ----------
        channel :
            Detector number, or ``None`` for the photons of all detectors
            (in record order; pair them with :attr:`photon_channel`).

        Returns
        -------
        ndarray
            T2 files: the time tags (unit :attr:`timetag_resolution`). T3
            files: laser pulse number times the laser period plus the delay
            after the pulse (to the lifetime resolution).
        """
        self._require_pulsed()
        if self.measurement_mode == "T2":
            times = self._all_events[:, 2]
        else:
            times = (self._all_events[:, 0] * self.rep_time
                     + self._all_raw_dtime * self.resolution)
        if channel is None:
            return times.copy()
        return times[self._all_channel == int(channel)]

    def _detector_list(self, channels: int | Sequence[int] | None) -> list[int]:
        """Detector numbers of ``channels`` (``None``: all), checked against the file."""
        if channels is None:
            return list(self.channels)
        values = [int(c) for c in np.atleast_1d(np.asarray(channels))]
        missing = [c for c in values if c not in self.channels]
        if missing:
            raise ValueError(f"{self.filepath.stem}: no photons on channel(s) {missing}; "
                             f"the file has channel(s) {list(self.channels)}.")
        return values

    def detector_trace(self, bintime: float = 50e-3,
                       channels: int | Sequence[int] | None = None) -> Signal:
        """Photons per time bin of the given detectors, from their exact arrival times.

        Unlike :meth:`time_trace` this neither uses nor changes the detector
        selection (:meth:`set_channels`) or the cached time trace, so it can
        be called alongside a lifetime analysis. It is the trace on which the
        regions of :meth:`g2` are chosen.

        Parameters
        ----------
        bintime :
            Width of each time bin in seconds; the bins start at 0 and cover
            the whole acquisition (the last one may be shorter).
        channels :
            Detector number(s) to count, or ``None`` for all detectors.

        Returns
        -------
        Signal
            ``x``: left edge of each bin (s), ``y``: photons in the bin.
            ``meta`` holds ``bintime``, ``bin_durations`` and ``channels``.
        """
        self._require_pulsed()
        if not np.isfinite(bintime) or bintime <= 0:
            raise ValueError("bintime must be a positive finite number.")
        chosen = self._detector_list(channels)
        times = self.photon_times()[np.isin(self._all_channel, chosen)]
        duration = float(self.acquisition_time)
        if times.size:
            duration = max(duration, float(times.max()))
        n_bins = max(int(np.ceil(duration / bintime)), 1)
        index = np.minimum((times / bintime).astype(np.int64), n_bins - 1)
        counts = np.bincount(index, minlength=n_bins).astype(float)
        left = np.arange(n_bins) * bintime
        durations = np.minimum(left + bintime, duration) - left
        return Signal(left, counts, label=f"{self.filepath.stem} detectors "
                                          f"{', '.join(map(str, chosen))}",
                      units={"x": "s", "y": "counts"},
                      meta={"bintime": float(bintime), "bin_durations": durations,
                            "channels": chosen})

    def g2(self, roi: dict[str, float] | None = None, *,
           x_min: float | None = None, x_max: float | None = None,
           y_min: float | None = None, y_max: float | None = None,
           bintime: float = 50e-3, channels: Sequence[int] = (1, 2),
           max_lag: float | None = None, lag_bin: float | None = None,
           period: float | None = None, delay: float | None = None,
           window: float | None = None,
           progress: Callable[[int, int], Any] | None = None) -> Signal:
        """Second-order correlation ``g2(τ)`` between two detectors (antibunching).

        Every photon pair of the two detectors within ``±max_lag`` is counted
        by the time difference ``t_b - t_a`` (Hanbury Brown–Twiss setup: one
        beam splitter, two detectors). Only T2 files are supported: their
        time tags have the full timing resolution of the hardware.

        Parameters
        ----------
        roi, x_min, x_max, y_min, y_max, bintime :
            Which photons are used: the time bins of :meth:`detector_trace`
            (width ``bintime``, photons of both detectors) inside the time
            window ``[x_min, x_max)`` whose counts lie in ``[y_min, y_max)``,
            as for :meth:`flid` (``roi`` overrides the explicit bounds). A
            count range selects e.g. the on state of a blinking emitter.
            All ``None``: the whole file.
        channels :
            The two detectors ``(a, b)``; positive lags mean the photon of
            ``b`` came later.
        max_lag :
            Largest lag in s. Default: 5.5 laser periods with a laser period
            (centre peak plus five side peaks on each side), else 500 ns.
        lag_bin :
            Width of one lag bin in s, rounded to whole time tags. Default:
            1/200 of the laser period, else 1 ns.
        period :
            Laser repetition period in s for the pulsed ``g2(0)``. Default:
            the period measured from the laser sync of the file; files
            without one (PicoHarp 300 T2) need it here. Without any period
            the ``g2(0)`` of continuous excitation is computed (see below).
        delay :
            Lag of the centre peak in s (detector and cable delays). ``None``:
            found from the peak comb with a period, 0 without one.
        window :
            Half-width in s of the area around each peak (pulsed; default half
            a period) or around ``delay`` (continuous; default half a lag
            bin, i.e. the bin at ``delay``).
        progress :
            Optional callback ``progress(done, total)``; raise from it to
            cancel.

        Returns
        -------
        Signal
            ``x``: lag (s, bin centres), ``y``: coincidences per lag bin.
            ``meta`` holds ``g2`` (the normalised correlation: coincidences
            divided by ``N_a N_b Δτ / T`` expected for uncorrelated light,
            with ``N`` photons per detector and ``T`` the selected duration),
            ``g2_0`` ± ``g2_0_err`` and ``method`` (``"pulsed"``: centre-peak
            area over the mean side-peak area, plus ``centre_area``,
            ``side_areas``, ``mean_side_area``, ``n_side_peaks``;
            ``"continuous"``: mean normalised ``g2`` within ``window`` of
            ``delay``), ``period``, ``delay``, ``window``, ``lag_bin``,
            ``max_lag``, ``photons`` per detector, ``rates`` (counts/s),
            ``duration``, ``selected_bins``, ``roi`` and ``bintime``.

        Notes
        -----
        Photons are taken from the selected time bins only, so pairs that
        straddle the edge of a selected bin into an unselected one are lost;
        this removes about ``max_lag / bintime`` of the coincidences at the
        edges and is negligible for bins much longer than the lags.
        """
        self._require_pulsed()
        if self.measurement_mode != "T2":
            raise ValueError(f"{self.filepath.stem}: g2 needs a T2 file (time tags of "
                             f"every photon); this is a {self.measurement_mode} file.")
        pair = self._detector_list(channels)
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("channels must name two different detectors, e.g. (1, 2).")
        tick = float(self.timetag_resolution)
        if period is None and self.has_lifetime and np.isfinite(self.rep_time):
            period = float(self.rep_time)
        if period is not None and (not np.isfinite(period) or period <= 0):
            raise ValueError("period must be a positive number of seconds or None.")
        if max_lag is None:
            max_lag = 5.5 * period if period is not None else 500e-9
        if lag_bin is None:
            lag_bin = period / 200.0 if period is not None else 1e-9
        if not np.isfinite(max_lag) or max_lag <= 0 or not np.isfinite(lag_bin) or lag_bin <= 0:
            raise ValueError("max_lag and lag_bin must be positive numbers of seconds.")
        width = max(int(round(lag_bin / tick)), 1)

        # Time bins of the selection, on the trace of both detectors.
        if roi is not None:
            x_min, x_max = roi.get("x_min"), roi.get("x_max")
            y_min, y_max = roi.get("y_min"), roi.get("y_max")
        trace = self.detector_trace(bintime, pair)
        left = np.asarray(trace.x, dtype=float)
        durations = np.asarray(trace.meta["bin_durations"], dtype=float)
        counts = np.asarray(trace.y, dtype=float)
        xmn = 0.0 if x_min is None else float(x_min)
        xmx = np.inf if x_max is None else float(x_max)
        ymn = -np.inf if y_min is None else float(y_min)
        ymx = np.inf if y_max is None else float(y_max)
        selected = ((left + durations > xmn) & (left < xmx)
                    & (counts >= ymn) & (counts < ymx))
        if not selected.any():
            raise ValueError("The selected region contains no time bins.")
        duration = float(durations[selected].sum())

        streams = []
        for channel in pair:
            times = self.photon_times(channel)
            index = np.minimum((times / bintime).astype(np.int64), selected.size - 1)
            streams.append(np.rint(times[selected[index]] / tick).astype(np.int64))
        n_a, n_b = (int(s.size) for s in streams)
        if n_a == 0 or n_b == 0:
            raise ValueError("One of the detectors has no photons in the selected region.")
        lag_ticks, coincidences = photon_correlation(
            streams[0], streams[1], max_lag / tick, width, progress=progress)
        lags = lag_ticks * tick
        expected = n_a * n_b * width * tick / duration
        g2 = coincidences / expected

        result: dict[str, Any]
        if period is not None:
            result = pulsed_g2_zero(lags, coincidences, period, delay=delay, window=window)
            result["method"] = "pulsed"
        else:
            centre = 0.0 if delay is None else float(delay)
            half = width * tick / 2.0 if window is None else float(window)
            inside = np.abs(lags - centre) <= half * (1 + 1e-9)
            if not inside.any():
                raise ValueError("The window around the delay holds no lag bin.")
            total = float(coincidences[inside].sum())
            value = float(np.mean(g2[inside]))
            result = {"method": "continuous", "g2_0": value,
                      "g2_0_err": value / np.sqrt(total) if total > 0
                      else 1.0 / (expected * inside.sum()),
                      "delay": centre, "window": half}
        bounds = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        meta = {**result, "g2": g2, "expected_per_bin": float(expected),
                "period": period, "lag_bin": width * tick, "max_lag": float(lags[-1]),
                "channels": pair, "photons": {pair[0]: n_a, pair[1]: n_b},
                "rates": {pair[0]: n_a / duration, pair[1]: n_b / duration},
                "duration": duration, "selected_bins": int(selected.sum()),
                "roi": bounds, "bintime": float(bintime)}
        print(f"{self.filepath.stem}: g2 of detectors {pair[0]} and {pair[1]} from "
              f"{n_a} + {n_b} photons in {duration:.4g} s: g2(0) = {result['g2_0']:.3f} "
              f"± {result['g2_0_err']:.3f} ({result['method']}).")
        return Signal(lags, coincidences.astype(float),
                      label=f"{self.filepath.stem} g2 ({pair[0]}×{pair[1]})",
                      units={"x": "s", "y": "coincidences"}, meta=meta)

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
        (:attr:`lifetime_seam_bin`); the lifetime fits leave it and the other
        seam bins out (:meth:`seam_bins`).
        All cached traces are discarded.
        """
        self._require_lifetime()
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

    def seam_bins(self) -> np.ndarray:
        """Lifetime bins that are not fully inside one laser period (the seam).

        Detected from the geometry of the file, not from count statistics:

        * bins past the laser period, which never fill;
        * the last bin of the period if the period is not a whole number of
          bins: it is only partly covered and collects fewer photons;
        * the bins after the last one that holds a photon anywhere in the
          file: there the real laser period is shorter than the header says
          (seen in legacy files whose sync only ran while the laser was on).

        Unshifted they sit at the end of the lifetime window; a lifetime
        shift (:meth:`set_lifetime_shift`) moves them into the decay, and
        this mask follows it. Every lifetime fit leaves them out by default
        (``exclude_seam``): a partly covered or empty bin does not follow the
        decay, and the bins before the rise (``prerise``) rely on the period.

        Returns
        -------
        ndarray of bool
            One entry per lifetime bin (:attr:`n_lt_bins`), ``True`` for a
            seam bin. All ``False`` when the laser period is unknown.
        """
        out = np.zeros(self.n_lt_bins, dtype=bool)
        resolution = getattr(self, "resolution", None)
        if self._period_ns() is None or not resolution:
            return out
        period_bins = self.rep_time / resolution
        wrap = int(np.ceil(period_bins - 1e-6))
        out = np.arange(self.n_lt_bins) >= wrap
        if wrap > self.n_lt_bins:
            return out                  # window shorter than the period: nothing wraps
        partial = self._partial_bin()
        if partial is not None:
            out[partial] = True
        raw = getattr(self, "_raw_dtime", None)
        if raw is not None and len(raw):
            # Raw bins that never filled, moved along with a lifetime shift.
            empty = np.arange(int(np.max(raw)) + 1, wrap)
            shift_bins = int(round(self.lifetime_shift_ns * 1e-9 / resolution))
            out[(empty - shift_bins) % wrap] = True
        return out

    def _partial_bin(self) -> int | None:
        """The partly covered last bin of the period (moved by a lifetime shift), if any."""
        resolution = getattr(self, "resolution", None)
        if self._period_ns() is None or not resolution:
            return None
        period_bins = self.rep_time / resolution
        wrap = int(np.ceil(period_bins - 1e-6))
        if wrap > self.n_lt_bins or period_bins - (wrap - 1) >= 0.99:
            return None
        return self.lifetime_seam_bin if self.lifetime_seam_bin is not None else wrap - 1

    def _seam_mask(self, size: int, exclude_seam: bool) -> np.ndarray:
        """:meth:`seam_bins` for a histogram of ``size`` bins, or all ``False``.

        Nothing is left out with ``exclude_seam=False`` or for a histogram
        that is not on this file's lifetime bins.
        """
        if not exclude_seam or size != self.n_lt_bins:
            return np.zeros(size, dtype=bool)
        return self.seam_bins()

    def _seam_report(self, t_ns: np.ndarray, seam: np.ndarray) -> dict[str, Any]:
        """Fit-result entries describing the seam bins left out (``t_ns`` per bin)."""
        partial = self._partial_bin() if seam.any() else None
        if partial is not None and not (partial < seam.size and seam[partial]):
            partial = None
        return {"seam_ns": None if partial is None else float(t_ns[partial]),
                "excluded_ns": np.asarray(t_ns, dtype=float)[seam],
                "seam_points": int(np.count_nonzero(seam))}

    def _channel_report(self) -> dict[str, Any]:
        """``{"channels": [...]}`` when detectors were selected, else empty."""
        if self.selected_channels is None:
            return {}
        return {"channels": list(self.selected_channels)}

    def _prerise_bins(self, t_ns: np.ndarray, window: np.ndarray, gap: float,
                      seam: np.ndarray | None = None) -> np.ndarray:
        """Bins before the rising edge that are fitted together with the tail.

        ``t_ns`` is the lifetime axis with the histogram maximum at 0,
        ``window`` the tail-fit window (its bins are not counted twice),
        ``gap`` the distance in ns the region keeps from the maximum, see
        :func:`~kpfm_pl.utils.prerise_mask`, and ``seam`` the seam bins to
        leave out (:meth:`seam_bins`).
        """
        if self._period_ns() is None:
            raise ValueError("Fitting the bins before the rise needs the laser period.")
        pre = prerise_mask(t_ns, gap) & ~np.asarray(window, dtype=bool)
        if seam is not None and seam.size == pre.size:
            pre &= ~seam
        if not pre.any():
            raise ValueError(
                f"No lifetime bins lie more than {gap:g} ns before the histogram maximum; "
                "there is nothing before the rise to fit (prerise).")
        return pre

    def _period_ns(self) -> float | None:
        """Laser period in ns for the incomplete-decay correction (``None`` if unknown)."""
        rep_time = getattr(self, "rep_time", None)
        if rep_time is None or not np.isfinite(rep_time) or rep_time <= 0:
            return None
        return float(rep_time) * 1e9

    def _require_pulsed(self) -> None:
        """Reject time-binned/lifetime analysis on continuous-wave files."""
        if self.is_cw:
            raise ValueError(
                f"{self.filepath.stem}: continuous-wave file (infinite sync "
                "period); time-binned and lifetime analysis require pulsed "
                "(T3) data."
            )

    def _require_lifetime(self) -> None:
        """Reject lifetime analysis on files whose photons have no lifetime."""
        self._require_pulsed()
        if not self.has_lifetime:
            raise ValueError(
                f"{self.filepath.stem}: T2 file without laser pulses (no laser sync on "
                "channel 0), so the photons have no lifetime information; only time "
                "traces, markers and triggers are available."
            )

    def _build_matrix(self, bintime: float) -> None:
        """Build the (time-bin x lifetime-bin) photon count matrix."""
        n_tt = int(np.ceil(self.acquisition_time / bintime))
        # events[:, 0] counts laser periods (T2 without a laser sync: time tags).
        sync_per_bin = bintime / (self._sync_unit or self.rep_time)
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
        if self.selected_channels is not None:
            meta["channels"] = list(self.selected_channels)

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
        self._require_lifetime()
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
        if self.selected_channels is not None:
            meta["channels"] = list(self.selected_channels)
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
        self._require_lifetime()
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
        end: float | None = None, weighting: str = DEFAULT_BIN_WEIGHTING,
        decay_correction: bool = True,
        irf: dict[str, Any] | None = None,
        prerise: bool = False, prerise_gap: float = 2.0,
        background_rate: float | None = None,
        exclude_seam: bool = True,
        progress: Callable[[int, int], Any] | None = None,
        **fit_kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit lifetimes per selected time bin; ``(taus, amps, amps_fit, fallback)``.

        All arrays keep every bin position. ``fallback`` marks the bins whose
        Poisson MLE did not converge and that hold the Poisson least-squares
        result instead (always ``False`` for the least-squares weightings).

        ``amps`` are the per-pulse amplitudes (corrected for the incomplete
        decay of earlier pulses when ``decay_correction``, see
        :class:`~kpfm_pl.fitting.MultiExpFit`'s ``period``), ``amps_fit`` the
        fitted ones. ``irf`` (see :meth:`lifetime_trace`) switches to the
        reconvolution fit. ``prerise``/``prerise_gap`` and ``background_rate``
        (tail fits) are described in :meth:`lifetime_trace`.
        """
        if irf is not None:
            return self._fit_time_bin_lifetimes_irf(
                sel, irf, n_exp=n_exp, bins=bins, mode=mode, start=start, end=end,
                weighting=weighting, decay_correction=decay_correction,
                exclude_seam=exclude_seam, progress=progress, **fit_kwargs)
        time, matrix, selected = sel["time"], sel["matrix"], sel["selected"]
        bintime = sel["bintime"]

        # Use the selected region to choose one common lifetime fit window, then
        # fit each selected row independently. Unselected/invalid/failed rows
        # remain NaN so their positions stay aligned with ``time``.
        lt_ns, t0_ns, mask = self._lifetime_window(
            matrix[selected, :], mode, fraction, smoothing, start, end)
        seam = self._seam_mask(mask.size, exclude_seam)
        pre = np.zeros(mask.shape, dtype=bool)
        if prerise:
            pre = self._prerise_bins(lt_ns - t0_ns, mask, prerise_gap, seam)
        if background_rate is not None and (not np.isfinite(background_rate)
                                            or background_rate < 0):
            raise ValueError("background_rate must be a finite, non-negative number or None.")

        group_starts = np.arange(0, matrix.shape[1], bins)
        if group_starts.size < 2:
            raise ValueError("bins must leave at least two lifetime groups for fitting.")
        group_sizes = np.minimum(bins, matrix.shape[1] - group_starts)
        binned_lt = np.add.reduceat(lt_ns, group_starts) / group_sizes
        # Only groups whose bins all lie in the window: a group straddling the
        # window start would add rising-edge bins at its mean time (too few
        # counts there, so the fitted lifetimes come out too long). A group
        # holding a seam bin is left out as a whole.
        outside = np.add.reduceat(seam.astype(int), group_starts) > 0
        mask = (np.add.reduceat(mask.astype(int), group_starts) == group_sizes) & ~outside
        t_rel = binned_lt - t0_ns
        if prerise:
            # A group lies before the rise only if all its bins do, and the
            # wrapped times need groups that are fully inside the period.
            whole = np.add.reduceat(pre.astype(int), group_starts) == group_sizes
            pre = whole & ~mask & ~outside
            if not pre.any():
                raise ValueError("bins leaves no whole lifetime group before the rise.")
            mask = mask | pre
            # Bins before the rise continue the decay of the previous pulse.
            t_rel = np.where(pre, t_rel + self._period_ns(), t_rel)
        t_fit = t_rel[mask]
        if t_fit.size < 2:
            raise ValueError("The selected lifetime fit window contains fewer than two bins.")
        binned_matrix = np.add.reduceat(matrix, group_starts, axis=1)

        taus = np.full((len(time), n_exp), np.nan, dtype=float)
        amps = np.full((len(time), n_exp), np.nan, dtype=float)
        amps_fit = np.full((len(time), n_exp), np.nan, dtype=float)
        fallback = np.zeros(len(time), dtype=bool)
        n_binned = len(mask)
        period_ns = self._period_ns() if decay_correction else None
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
            # The offset must be in the units of y_fit: counts / (norm * ymax).
            scale = norm * ymax
            if background_rate is not None:
                offset = float(background_rate) * bintime * bins / scale
            else:
                offset = self.dark_counts * bintime / n_binned / scale
            bin_kwargs = _poisson_weights(weighting, lt_bin[mask], scale, fit_kwargs)
            fit = MultiExpFit(t_fit, y_fit, n_exp, offset, period=period_ns, **bin_kwargs)
            if fit.success:
                params = fit.parameters()
                fitted_taus = np.asarray(params["tau"], dtype=float)
                fitted_amps = np.asarray(params["A"], dtype=float)
                raw_amps = np.asarray(params.get("A_fit", fitted_amps), dtype=float)
                if (fitted_taus.shape == fitted_amps.shape == (n_exp,)
                        and np.all(np.isfinite(fitted_taus))
                        and np.all(np.isfinite(fitted_amps))):
                    taus[index] = fitted_taus
                    amps[index] = fitted_amps
                    amps_fit[index] = raw_amps
                    fallback[index] = getattr(fit, "mle_converged", None) is False
        if progress is not None:
            progress(rows.size, rows.size)
        return taus, amps, amps_fit, fallback

    def _fit_time_bin_lifetimes_irf(
        self, sel: dict[str, Any], irf: dict[str, Any], *, n_exp: int, bins: int,
        mode: str, start: float, end: float | None, weighting: str,
        decay_correction: bool, exclude_seam: bool = True,
        progress: Callable[[int, int], Any] | None = None,
        **fit_kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reconvolution fit of every selected time bin (see :meth:`lifetime_trace`)."""
        time, matrix, selected = sel["time"], sel["matrix"], sel["selected"]
        bintime = sel["bintime"]
        wrap = self._period_grid()
        seam = self._seam_mask(self.n_lt_bins, exclude_seam)[:wrap]
        irf_full, _ = prepare_irf(self.irf_histogram(irf["irf"])[:wrap],
                                  irf.get("background", "auto"))
        # Group the lifetime bins of one period by ``bins``.
        starts = np.arange(0, wrap, bins)
        if starts.size < 2 * n_exp + 3:
            raise ValueError("bins leaves too few lifetime groups for the reconvolution fit.")
        sizes = np.minimum(bins, wrap - starts)
        irf_binned = np.add.reduceat(irf_full, starts)
        binned = np.add.reduceat(np.asarray(matrix[:, :wrap], dtype=float), starts, axis=1)
        lt_ns = self.lt_time[:wrap] * 1e9
        t_binned = np.add.reduceat(lt_ns, starts) / sizes
        summed = binned[selected].sum(axis=0)
        t_rel = t_binned - t_binned[int(np.argmax(summed))]
        if mode == "manual":
            mask = fit_window_mask(t_rel, summed, "manual", start=start, end=end)
        else:
            mask = np.ones(starts.size, dtype=bool)
        # A group holding a seam bin is left out as a whole.
        mask &= ~(np.add.reduceat(seam.astype(int), starts) > 0)
        if mask.sum() < 2 * n_exp + 3:
            raise ValueError("The fit window holds too few bins for the reconvolution fit.")
        dt = self.resolution * 1e9 * bins
        shift = irf.get("shift")
        rate = irf.get("background_rate")
        fraction = float(irf.get("scatter_fraction") or 0.0)

        taus = np.full((len(time), n_exp), np.nan, dtype=float)
        amps = np.full((len(time), n_exp), np.nan, dtype=float)
        fallback = np.zeros(len(time), dtype=bool)
        rows = np.flatnonzero(selected)
        for done, index in enumerate(rows):
            if progress is not None:
                progress(done, rows.size)
            counts = binned[index]
            norm = float(np.max(counts))
            if not np.isfinite(norm) or norm == 0:
                continue
            if rate is not None:
                background = float(rate) * bintime * bins
            else:
                background = self.dark_counts * bintime / self.n_lt_bins * bins
            bin_kwargs = _poisson_weights(weighting, counts[mask], norm, fit_kwargs)
            try:
                fit = IRFReconvolutionFit(
                    irf_binned, counts / norm, n_exp, dt, fit_mask=mask,
                    periodic=decay_correction, background=background / norm,
                    scatter_counts=fraction * float(np.sum(counts)) / norm, shift=shift,
                    irf_background=None, **bin_kwargs)
            except ValueError:
                continue
            if fit.success:
                params = fit.parameters()
                fitted_taus = np.asarray(params["tau"], dtype=float)
                fitted_amps = np.asarray(params["A"], dtype=float)
                if np.all(np.isfinite(fitted_taus)) and np.all(np.isfinite(fitted_amps)):
                    taus[index] = fitted_taus
                    amps[index] = fitted_amps
                    fallback[index] = getattr(fit, "mle_converged", None) is False
        if progress is not None:
            progress(rows.size, rows.size)
        # The periodic kernel already gives per-pulse amplitudes; nothing to undo.
        return taus, amps, amps.copy(), fallback

    def _validate_fit_args(self, n_exp: int, bins: int, weighting: str,
                           fit_kwargs: dict[str, Any],
                           mean_lifetime: str = "intensity") -> str:
        """Validate the per-bin fit arguments; returns the weighting actually used."""
        # Validate before the (slow) loop rather than at the first fit.
        weighting = _resolve_weighting(weighting, fit_kwargs, self.filepath.stem)
        _poisson_weights(weighting, np.ones(1), 1.0, fit_kwargs)
        _check_mean_lifetime(mean_lifetime)
        if (isinstance(n_exp, (bool, np.bool_))
                or not isinstance(n_exp, (int, np.integer)) or n_exp < 1):
            raise ValueError("n_exp must be a positive integer.")
        if (isinstance(bins, (bool, np.bool_))
                or not isinstance(bins, (int, np.integer)) or bins < 1):
            raise ValueError("bins must be a positive integer.")
        return weighting

    def _trace_meta(self, sel: dict[str, Any], method: str) -> dict[str, Any]:
        meta = {"bintime": sel["bintime"], "method": method, "roi": sel["bounds"],
                "counts": sel["counts"], "laser_pulse": sel["laser_pulse"],
                "lifetime_shift_ns": self.lifetime_shift_ns,
                "count_check": sel["count_check"]}
        if self.selected_channels is not None:
            meta["channels"] = list(self.selected_channels)
        return meta

    def lifetime_trace(self, roi: dict[str, float] | None = None, *,
                       x_min: float | None = None, x_max: float | None = None,
                       y_min: float | None = None, y_max: float | None = None,
                       bintime: float = 50e-3, n_exp: int = 1, bins: int = 1,
                       mode: str = "fraction", fraction: float = 0.95,
                       smoothing: float = 2.0, start: float = 0.0,
                       end: float | None = None,
                       laser_pulse: bool | None = None,
                       weighting: str = DEFAULT_BIN_WEIGHTING,
                       min_counts: float | None = 100, coverage: float = 0.9,
                       auto_bintime: bool = False,
                       mean_lifetime: str = "intensity",
                       decay_correction: bool = True,
                       irf: PTU | Signal | np.ndarray | None = None,
                       irf_shift: float | None = None,
                       background_rate: float | None = None,
                       scatter_fraction: float | None = None,
                       irf_background: float | str | None = "auto",
                       prerise: bool = False, prerise_gap: float = 2.0,
                       exclude_seam: bool = True,
                       progress: Callable[[int, int], Any] | None = None,
                       **fit_kwargs) -> Signal:
        """Return mean fitted lifetime versus acquisition time.

        The returned ``Signal.x`` contains the time-bin positions and
        ``Signal.y`` contains the mean of the fitted exponential lifetimes in
        ns (equal to ``tau`` for ``n_exp=1``): intensity-weighted
        ``sum(A_i * tau_i**2) / sum(A_i * tau_i)`` with
        ``mean_lifetime="intensity"`` (default), amplitude-weighted
        ``sum(A_i * tau_i) / sum(A_i)`` with ``"amplitude"``. There is exactly
        one value per time bin. A value is ``NaN`` when the bin is outside the
        requested intensity ROI, contains no usable photons, its fit fails, or
        all its amplitudes are zero. Individual fitted components are
        available as ``signal.meta["taus"]`` and ``signal.meta["amplitudes"]``
        with shape ``(time bins, n_exp)``. With ``decay_correction`` (default)
        the amplitudes (and so the mean lifetime) are per-pulse values,
        corrected for the incomplete decay of earlier laser pulses as in
        :meth:`fit_lifetime`; the uncorrected ones are in
        ``signal.meta["amplitudes_fit"]``.
        By default, the trace inherits the ``laser_pulse`` mode of the most
        recent :meth:`time_trace` call.  Pass the option explicitly when no
        time trace has been generated first.

        ``weighting`` (``"poisson"``, the default, ``"mle"`` or ``"none"``) is
        applied to every per-bin fit as in :meth:`fit_lifetime`. Poisson
        least squares is fast and close to the maximum likelihood from about
        100 photons per bin; the Poisson MLE (``"mle"``) takes several times
        longer. Bins where the MLE does not converge get the Poisson
        least-squares result instead: ``meta["mle_fallback"]`` marks them and
        one :class:`~kpfm_pl.fitting.MLEFallbackWarning` gives their number.
        ``meta["weighting"]`` is the weighting used. ``progress`` is an optional
        callback ``progress(done, total)`` called before each per-bin fit and
        once at the end; raise an exception from it to abort the computation
        (see :meth:`flid`). ``min_counts``, ``coverage`` and ``auto_bintime``
        check the photons per time bin, see :meth:`flid`.

        With ``irf`` every bin is fitted by IRF reconvolution as in
        :meth:`fit_lifetime` (whole period unless ``mode="manual"``;
        ``decay_correction`` selects the periodic kernel). Per-bin fits have
        few photons, so take the shared parameters from a fit of the summed
        histogram: ``irf_shift`` fixes the IRF shift (ns; ``None`` fits it in
        every bin), ``background_rate`` the background in counts per second
        and lifetime bin (e.g. ``background / acquisition_duration`` of that
        fit; default: the dark counts) and ``scatter_fraction`` the scattered
        light as a fraction of each bin's photons (e.g. ``scatter /
        total counts``; default 0).

        Without ``irf`` (tail fits), ``prerise`` also fits the bins that lie
        more than ``prerise_gap`` ns before the histogram maximum, at their
        time since the previous pulse, as in :meth:`fit_lifetime`. The
        constant offset of a bin is never fitted (too few photons): it is
        ``background_rate`` (counts per second and lifetime bin, e.g. from
        ``fit_lifetime(..., fit_background=True)`` of the summed histogram)
        or, by default, the dark counts.

        ``exclude_seam`` (default) leaves the seam bins out of every per-bin
        fit (:meth:`seam_bins`; with ``bins > 1`` the whole group holding
        one), as in :meth:`fit_lifetime`. ``meta["seam_points"]`` is their
        number.
        """
        weighting = self._validate_fit_args(n_exp, bins, weighting, fit_kwargs, mean_lifetime)
        irf_options = None
        if irf is not None:
            irf_options = {"irf": irf, "shift": irf_shift, "background_rate": background_rate,
                           "scatter_fraction": scatter_fraction, "background": irf_background}
            self.irf_histogram(irf)        # validate before the (slow) selection and loop
            self._period_grid()
        sel = self._select_time_bins(
            roi, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            bintime=bintime, laser_pulse=laser_pulse, min_counts=min_counts,
            coverage=coverage, auto_bintime=auto_bintime,
        )
        taus, amps, amps_fit, fallback = self._fit_time_bin_lifetimes(
            sel, n_exp=n_exp, bins=bins, mode=mode, fraction=fraction,
            smoothing=smoothing, start=start, end=end, weighting=weighting,
            decay_correction=decay_correction, irf=irf_options, prerise=prerise,
            prerise_gap=prerise_gap, background_rate=background_rate,
            exclude_seam=exclude_seam, progress=progress,
            **fit_kwargs,
        )
        tail_prerise = bool(prerise) and irf is None
        n_seam = int(np.count_nonzero(self._seam_mask(self.n_lt_bins, exclude_seam)))
        n_fallback = int(np.count_nonzero(fallback))
        if n_fallback:
            n_fitted = int(np.count_nonzero(np.isfinite(taus[:, 0])))
            warnings.warn(
                f"{self.filepath.stem}: the Poisson maximum-likelihood fit did not converge "
                f"in {n_fallback} of {n_fitted} fitted time bins; the Poisson least-squares "
                "result is used there (meta['mle_fallback']).",
                MLEFallbackWarning, stacklevel=2)
        return Signal(
            sel["time"], _weighted_mean_lifetime(taus, amps, mean_lifetime),
            label=f"{self.filepath.stem} mean lifetime",
            units={"x": "s", "y": "ns"},
            meta={**self._trace_meta(sel, "fit"), "n_exp": n_exp, "lt_bins": bins,
                  "taus": taus, "amplitudes": amps, "amplitudes_fit": amps_fit,
                  "weighting": weighting, "mle_fallback": fallback,
                  "mean_lifetime": mean_lifetime,
                  "decay_correction": bool(decay_correction),
                  "reconvolution": irf is not None, "irf_shift_ns": irf_shift,
                  "prerise": tail_prerise,
                  "prerise_gap_ns": float(prerise_gap) if tail_prerise else None,
                  "background_rate": background_rate,
                  "exclude_seam": bool(exclude_seam), "seam_points": n_seam},
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
                           exclude_seam: bool = True,
                           progress: Callable[[int, int], Any] | None = None) -> Signal:
        """Lifetime per time bin from the mean photon arrival time, without fitting.

        A fast sanity check of :meth:`lifetime_trace`: same arguments, same
        time bins and the same fit window (chosen on the summed histogram of
        the selected bins). For each bin, the expected dark counts per lifetime
        channel (``dark_counts * bintime / n_lt_bins``) are subtracted and the
        mean delay of the remaining photons after the window start is taken.

        Observed without limit, this mean is exactly ``tau`` for one
        exponential, wherever the window starts. For a decay
        ``sum(A_i exp(-t/tau_i))`` it is the intensity-weighted mean lifetime
        of the decay at the window start ``t0``:
        ``sum(a_i tau_i**2) / sum(a_i tau_i)`` with ``a_i = A_i exp(-t0/tau_i)``,
        longer than the ``tau_int`` of the fits (amplitudes at ``t = 0``, the
        histogram maximum) for ``t0 > 0``, since the short components have
        already faded. The window of length ``T`` cuts off late photons, so
        the mean is too short: ``tau - T / (exp(T / tau) - 1)`` for one
        exponential. ``truncation_correction=True`` (default) inverts this
        relation (:func:`~kpfm_pl.utils.lifetime_from_truncated_mean`); it is
        exact for one exponential and approximate for several. Without dark-count
        subtraction (``dark_counts`` set wrongly) the flat background pulls the
        mean towards ``T / 2``. ``exclude_seam`` (default) leaves the seam
        bins (:meth:`seam_bins`) out of the mean.

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
        channels = np.flatnonzero(mask & ~self._seam_mask(mask.size, exclude_seam))
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
                  "truncation_correction": bool(truncation_correction),
                  "exclude_seam": bool(exclude_seam)},
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
             isolated: bool = False, weighting: str = DEFAULT_BIN_WEIGHTING,
             method: str = "fit", truncation_correction: bool = True,
             min_counts: float | None = 100, coverage: float = 0.9,
             auto_bintime: bool = False,
             mean_lifetime: str = "intensity", decay_correction: bool = True,
             irf: PTU | Signal | np.ndarray | None = None,
             irf_shift: float | None = None, background_rate: float | None = None,
             scatter_fraction: float | None = None,
             prerise: bool = False, prerise_gap: float = 2.0,
             exclude_seam: bool = True,
             progress: Callable[[int, int], Any] | None = None,
             **fit_kwargs) -> Image | list[Image]:
        """Compute a FLID (Fluorescence-Lifetime-Intensity-Distribution) map.

        Each time bin of the (optionally) ROI-restricted matrix is fitted with
        ``n_exp`` exponentials; the resulting ``(lifetime, counts)`` pairs are
        binned into a 2-D histogram (lifetime on x, counts on y). Unless
        ``isolated``, the lifetime of a bin is the mean of its fitted
        components (equal to ``tau`` for ``n_exp=1``): intensity-weighted
        ``sum(A_i * tau_i**2) / sum(A_i * tau_i)`` by default, or
        amplitude-weighted with ``mean_lifetime="amplitude"``.

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
            otherwise return a single map of the per-bin mean lifetime.
        weighting :
            ``"poisson"`` (default, Poisson-weighted least squares), ``"mle"``
            (Poisson maximum likelihood, several times slower) or ``"none"``
            for the per-bin fits, see :meth:`lifetime_trace`.
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
        mean_lifetime :
            ``"intensity"`` (default) or ``"amplitude"``: the mean lifetime of
            a bin, see :meth:`lifetime_trace`. Only for ``method="fit"``; the
            mean arrival time always corresponds to the intensity-weighted mean.
        decay_correction :
            Correct the per-bin amplitudes for the incomplete decay of earlier
            laser pulses (default), see :meth:`fit_lifetime`.
        irf, irf_shift, background_rate, scatter_fraction :
            Per-bin IRF reconvolution, see :meth:`lifetime_trace` (``method="fit"``
            only). ``background_rate`` also sets the constant offset of the
            per-bin tail fits.
        prerise, prerise_gap :
            Also fit the bins before the rise in the per-bin tail fits, see
            :meth:`lifetime_trace` (``method="fit"`` only).
        exclude_seam :
            Leave the seam bins (:meth:`seam_bins`) out of every per-bin fit
            or mean arrival time (default).
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
                  "auto_bintime": auto_bintime, "exclude_seam": exclude_seam,
                  "progress": progress, **window}
        if method == "fit":
            trace = self.lifetime_trace(roi, n_exp=n_exp, bins=bins, weighting=weighting,
                                        mean_lifetime=mean_lifetime,
                                        decay_correction=decay_correction,
                                        irf=irf, irf_shift=irf_shift,
                                        background_rate=background_rate,
                                        scatter_fraction=scatter_fraction,
                                        prerise=prerise, prerise_gap=prerise_gap,
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
        self._require_lifetime()
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
                     normalize: bool = False, weighting: str = DEFAULT_WEIGHTING,
                     diagnostics: dict[str, float] | None = None,
                     mean_lifetime: str = "intensity", decay_correction: bool = True,
                     irf: PTU | Signal | np.ndarray | None = None,
                     fit_background: bool = False, fit_scatter: bool = False,
                     irf_shift: float | None = None,
                     irf_background: float | str | None = "auto",
                     prerise: bool = False, prerise_gap: float = 2.0,
                     exclude_seam: bool = True,
                     **fit_kwargs) -> dict[str, Any]:
        """Fit a lifetime ``histogram`` with ``n_exp`` exponentials.

        The fit window left bound is chosen by ``mode``: ``"fraction"`` (rising
        edge), ``"inflection"`` or ``"manual"`` (``start``/``end`` in ns). Returns
        the :meth:`~kpfm_pl.fitting.MultiExpFit.to_dict` payload augmented with the
        time/intensity arrays actually fitted.

        ``weighting="mle"`` (default) fits by Poisson maximum likelihood,
        which stays unbiased at low photon numbers (works with ``normalize``
        too). ``weighting="poisson"`` is least squares with every histogram
        bin weighted by its photon counting noise
        ``1 / sqrt(max(counts, 1))``; a robust ``loss``'s ``f_scale`` is then
        in units of that standard deviation. ``"none"`` is unweighted least
        squares. The maximum likelihood has no robust loss and sets its own
        weights: with a robust ``loss`` the fit falls back to ``"poisson"``,
        with explicit ``weights`` to ``"none"``, and says so in a
        ``RuntimeWarning``. An MLE that does not converge also falls back to
        the Poisson least-squares fit (``out["mle_converged"]`` is then
        ``False``, with an :class:`~kpfm_pl.fitting.MLEFallbackWarning`).
        ``out["weighting"]`` is the weighting actually used. Either way,
        ``out["quality"]``
        additionally holds the Poisson goodness of fit from
        :func:`~kpfm_pl.fitting.poisson_chi_squared` (``reduced_chi_squared``,
        ``chi_squared``, ``dof``, ``deviance``, ``weighted_residuals`` on
        ``t_fit``), computed on the raw counts even when ``normalize`` is set,
        and the Bayesian information criterion ``bic`` (from the chi-squared,
        or from the deviance for ``"mle"``) for comparing fits of the same
        histogram and window with different ``n_exp``.

        With ``decay_correction`` (default) the amplitudes ``A`` (and every
        quantity derived from them: ``tau_int``, ``tau_amp``, ``f_amp``,
        ``f_int``) are per-pulse values: the fitted ``A_fit`` include the
        photons of all earlier laser pulses that have not decayed yet,
        ``A = A_fit * (1 - exp(-T/tau))`` with the laser period ``T``. The
        lifetimes need no correction. ``decay_correction=False`` reports the
        fitted amplitudes unchanged.

        ``out["tau_mean"]``/``out["tau_mean_err"]`` repeat the mean lifetime
        selected by ``mean_lifetime``: ``"intensity"`` (``tau_int``, default)
        or ``"amplitude"`` (``tau_amp``); both are always returned as well.

        ``out["warnings"]`` lists signs of too many components (amplitude
        compatible with zero, merged lifetimes, parameter at a bound), see
        :func:`~kpfm_pl.fitting.fit_warnings`; ``diagnostics`` overrides its
        thresholds (e.g. ``{"min_fraction": 0.02}``).

        **Background and the bins before the rise (tail fit).** By default
        the constant offset is fixed at the dark counts. ``fit_background``
        fits it instead (``out["background"]`` ± ``out["background_err"]``,
        counts per bin; one more parameter in the BIC), for a pedestal the
        dark counts do not cover (afterpulsing, stray light, emission much
        slower than the laser period). ``prerise`` also fits the bins that
        lie more than ``prerise_gap`` ns before the histogram maximum. With
        periodic excitation they hold the tail of the previous pulse, so
        they enter the fit at their time since that pulse, ``t + T``
        (``out["t_model"]``; ``out["t_fit"]`` keeps the plot position), and
        extend the observed decay to the whole laser period. The first
        ``out["prerise_points"]`` fitted points are those bins. A
        fitted background and a lifetime above half the laser period are
        barely separable, which ``out["warnings"]`` reports
        (``diagnostics={"period_fraction": ...}``). ``prerise`` is not used
        with ``irf``: the reconvolution fits the whole period anyway.

        **Seam bins.** ``exclude_seam`` (default) leaves out the lifetime
        bins that are not fully inside one laser period (:meth:`seam_bins`:
        bins past the period, its partly covered last bin, bins that never
        fill; they follow a lifetime shift), in the tail fit, the bins before
        the rise and the reconvolution alike. ``out["excluded_ns"]`` holds
        their positions on ``shifted_histogram_x``, ``out["seam_points"]``
        their number and ``out["seam_ns"]`` the partly covered bin (``None``
        without one). ``exclude_seam=False`` fits them like any other bin.
        Only histograms on this file's lifetime bins have seam bins.

        **IRF reconvolution.** With ``irf`` (a :class:`PTU` of the recorded
        instrument response, its lifetime histogram as :class:`Signal`, or an
        array on the same lifetime bins) the model is convolved with the IRF
        and fitted over the whole laser period, including the rising edge
        (:class:`~kpfm_pl.fitting.IRFReconvolutionFit`). The IRF must have the
        same TCSPC resolution and laser period, and the lifetime window must
        cover a whole period. ``mode="fraction"``/``"inflection"`` then fit
        the whole period (except the seam bins), ``"manual"`` the
        bins between ``start`` and ``end``. The IRF shift ``δ`` is fitted
        (``irf_shift`` fixes it, in ns); ``fit_background`` also fits the
        constant background ``B`` (otherwise fixed from the dark counts) and
        ``fit_scatter`` a scattered-light contribution ``s`` (counts shaped
        like the IRF). ``decay_correction`` selects the periodic kernel (decay
        of all earlier pulses) or a single pulse; the amplitudes are per-pulse
        values of the IRF-free decay. ``irf_background`` is subtracted from
        the IRF (``"auto"``: the mean of its floor bins). ``normalize`` is not
        available.
        The result adds ``shift_ns``/``shift_err_ns``, ``background``,
        ``scatter`` (with errors), ``irf_fwhm_ns``, ``irf_shifted`` and
        ``full_curve`` (on ``irf_t_ns``, the period's time axis) and
        ``reconvolution: True``; ``quality["n_params"]`` counts δ, B and s.
        """
        self._require_lifetime()
        _check_mean_lifetime(mean_lifetime)
        weighting = _resolve_weighting(weighting, fit_kwargs, self.filepath.stem)
        if irf is not None:
            if normalize:
                raise ValueError("normalize is not available with an IRF: the reconvolution "
                                 "fits raw counts.")
            return self._fit_lifetime_irf(
                histogram, n_exp, irf, mode=mode, start=start, end=end,
                weighting=weighting, diagnostics=diagnostics, mean_lifetime=mean_lifetime,
                decay_correction=decay_correction, fit_background=fit_background,
                fit_scatter=fit_scatter, irf_shift=irf_shift, irf_background=irf_background,
                exclude_seam=exclude_seam, **fit_kwargs)
        y = np.asarray(histogram["y"], dtype=float)
        t0 = histogram["x"][np.argmax(y)]
        t_ns = (histogram["x"] - t0) * 1e9

        mask = fit_window_mask(t_ns, y, mode, fraction=fraction,
                               smoothing=smoothing, start=start, end=end)
        # Bins not fully inside one laser period do not follow the decay.
        seam = self._seam_mask(y.size, exclude_seam)
        mask = mask & ~seam
        tail = mask
        pre = np.zeros(y.size, dtype=bool)
        t_model = np.asarray(t_ns, dtype=float)
        if prerise:
            # Bins before the rise continue the decay of the previous pulse.
            pre = self._prerise_bins(t_model, tail, prerise_gap, seam)
            t_model = np.where(pre, t_model + self._period_ns(), t_model)
            mask = tail | pre

        t_fit, y_fit = t_ns[mask], y[mask]
        norm = 1.0
        if normalize and np.max(y_fit) > 0:
            norm = np.max(y_fit)
            y_fit = y_fit / norm
        offset = self._dark_offset(histogram) / norm

        fit_kwargs = _poisson_weights(weighting, y[mask], norm, fit_kwargs)
        fit = MultiExpFit(t_model[mask], y_fit, n_exp, offset,
                          period=self._period_ns() if decay_correction else None,
                          fit_offset=fit_background, **fit_kwargs)
        weighting = self._mle_outcome(fit, weighting)
        out = fit.to_dict()
        self._finish_fit(out, fit, y[mask], norm, 2 * n_exp + int(bool(fit_background)),
                         weighting=weighting, diagnostics=diagnostics,
                         mean_lifetime=mean_lifetime, decay_correction=decay_correction,
                         period=self._period_ns())
        out.update({"shifted_histogram_x": t_ns, "t_fit": t_fit, "y_fit": y_fit,
                    "t_model": t_model[mask],
                    "left_bound_ns": float(t_ns[tail][0]), "weighting": weighting,
                    "lifetime_shift_ns": self.lifetime_shift_ns,
                    "exclude_seam": bool(exclude_seam), **self._seam_report(t_ns, seam),
                    **self._channel_report(),
                    "fit_background": bool(fit_background),
                    "background": float(out.get("offset", offset)) * norm,
                    "background_err": float(out.get("offset_err", 0.0)) * norm,
                    "prerise": bool(prerise), "prerise_points": int(np.count_nonzero(pre)),
                    "prerise_gap_ns": float(prerise_gap) if prerise else None})
        return out

    def _mle_outcome(self, fit: MultiExpFit, weighting: str) -> str:
        """The weighting a finished fit used; warns when the MLE fell back.

        A Poisson MLE that did not converge returns the Poisson least-squares
        fit (:class:`~kpfm_pl.fitting.MultiExpFit`), so the result is
        reported as ``"poisson"``: its BIC comes from the chi-squared and it
        is not compared with MLE fits.
        """
        if weighting == "mle" and getattr(fit, "mle_converged", None) is False:
            warnings.warn(
                f"{self.filepath.stem}: the Poisson maximum-likelihood fit did not converge; "
                "the Poisson least-squares fit (weighting='poisson') is reported instead. "
                "Check the fit window and the background.", MLEFallbackWarning, stacklevel=3)
            return "poisson"
        return weighting

    def _dark_offset(self, histogram: Signal) -> float:
        """Dark counts per lifetime bin over the histogram's acquisition duration."""
        duration = histogram["meta"].get("acquisition_duration")
        if self.dark_counts and duration is None:
            raise ValueError(
                "Histogram acquisition duration is required for dark-count correction. "
                "Build it with lifetime_histogram(matrix, bintime=...)."
            )
        return self.dark_counts * float(duration or 0.0) / self.n_lt_bins

    @staticmethod
    def _finish_fit(out: dict[str, Any], fit: MultiExpFit, counts: np.ndarray, norm: float,
                    n_params: int, *, weighting: str, diagnostics: dict[str, float] | None,
                    mean_lifetime: str, decay_correction: bool,
                    period: float | None = None) -> None:
        """Add the chosen mean lifetime, Poisson quality, BIC and warnings to ``out``.

        ``period`` (ns) is the laser period for the warning about a fitted
        background next to a long lifetime (also without ``decay_correction``).
        """
        key = "tau_int" if mean_lifetime == "intensity" else "tau_amp"
        if key in out:
            out["tau_mean"], out["tau_mean_err"] = out[key], out[f"{key}_err"]
        out["mean_lifetime"] = mean_lifetime
        out["decay_correction"] = bool(decay_correction)
        if "curve" in out:
            out.setdefault("quality", {}).update(
                poisson_chi_squared(counts, np.asarray(out["curve"]) * norm, n_params,
                                    bic_from="deviance" if weighting == "mle"
                                    else "chi_squared"))
        converged = out.get("success") and "curve" in out
        thresholds = dict(diagnostics or {})
        if period is not None:
            thresholds.setdefault("period", period)
        out["warnings"] = fit.diagnostics(**thresholds) if converged else []

    # ---------------------------------------------------------- IRF reconvolution
    def irf_histogram(self, irf: PTU | Signal | np.ndarray) -> np.ndarray:
        """Lifetime histogram of an IRF on this file's lifetime bins (validated).

        ``irf`` is a :class:`PTU` of the recorded IRF (all its photons are
        used), its lifetime histogram (:class:`Signal`) or an array. A
        :class:`PTU` must have the same TCSPC resolution and laser period;
        every input must have :attr:`n_lt_bins` entries.
        """
        self._require_lifetime()
        if isinstance(irf, PTU):
            irf._require_lifetime()
            name = irf.filepath.name
            if not np.isclose(irf.resolution, self.resolution, rtol=1e-6, atol=0.0):
                raise ValueError(
                    f"IRF {name}: TCSPC resolution {irf.resolution * 1e12:.4g} ps differs from "
                    f"{self.resolution * 1e12:.4g} ps of {self.filepath.name}.")
            if not np.isclose(irf.rep_time, self.rep_time, rtol=1e-4, atol=0.0):
                raise ValueError(
                    f"IRF {name}: laser period {irf.rep_time * 1e9:.4g} ns differs from "
                    f"{self.rep_time * 1e9:.4g} ns of {self.filepath.name}.")
            counts = np.bincount(irf.events[:, 1].astype(int),
                                 minlength=irf.n_lt_bins)[:irf.n_lt_bins]
        elif isinstance(irf, Signal):
            counts = irf.y
        else:
            counts = irf
        counts = np.asarray(counts, dtype=float)
        if counts.shape != (self.n_lt_bins,):
            raise ValueError(f"The IRF histogram has {counts.size} bins; "
                             f"{self.filepath.name} has {self.n_lt_bins} lifetime bins.")
        return counts

    def _period_grid(self) -> int:
        """Number of lifetime bins of one laser period (the window must cover it)."""
        period_bins = self.rep_time / self.resolution
        wrap = int(np.ceil(period_bins - 1e-6))
        if self.n_lt_bins < wrap:
            raise ValueError(
                f"{self.filepath.stem}: the lifetime window ({self.n_lt_bins} bins, "
                f"{self.n_lt_bins * self.resolution * 1e9:.4g} ns) is shorter than one laser "
                f"period ({self.rep_time * 1e9:.4g} ns). IRF reconvolution needs the whole "
                "period; record with a coarser TCSPC resolution.")
        return wrap

    def _fit_lifetime_irf(self, histogram: Signal, n_exp: int, irf, *, mode: str,
                          start: float, end: float | None, weighting: str,
                          diagnostics: dict[str, float] | None, mean_lifetime: str,
                          decay_correction: bool, fit_background: bool, fit_scatter: bool,
                          irf_shift: float | None, irf_background: float | str | None,
                          exclude_seam: bool = True, **fit_kwargs) -> dict[str, Any]:
        y = np.asarray(histogram["y"], dtype=float)
        if y.size != self.n_lt_bins:
            raise ValueError("IRF reconvolution needs a lifetime histogram of this file "
                             f"({self.n_lt_bins} bins).")
        irf_counts = self.irf_histogram(irf)
        wrap = self._period_grid()
        seam = self._seam_mask(y.size, exclude_seam)
        x = np.asarray(histogram["x"], dtype=float)
        t_ns = (x - x[int(np.argmax(y))]) * 1e9
        t_p, y_p = t_ns[:wrap], y[:wrap]
        if mode == "manual":
            mask = fit_window_mask(t_p, y_p, "manual", start=start, end=end)
        else:
            mask = np.ones(wrap, dtype=bool)
        mask &= ~seam[:wrap]
        if mask.sum() < 2 * n_exp + 3:
            raise ValueError("The fit window holds too few bins for the reconvolution fit.")
        fit_kwargs = _poisson_weights(weighting, y_p[mask], 1.0, fit_kwargs)
        fit = IRFReconvolutionFit(
            irf_counts[:wrap], y_p, n_exp, self.resolution * 1e9, fit_mask=mask, t=t_p,
            periodic=decay_correction, background=self._dark_offset(histogram),
            fit_background=fit_background, scatter=fit_scatter, shift=irf_shift,
            irf_background=irf_background, **fit_kwargs)
        weighting = self._mle_outcome(fit, weighting)
        out = fit.to_dict()
        self._finish_fit(out, fit, y_p[mask], 1.0, fit._n_params(), weighting=weighting,
                         diagnostics=diagnostics, mean_lifetime=mean_lifetime,
                         decay_correction=decay_correction, period=self._period_ns())
        out.update({"shifted_histogram_x": t_ns, "t_fit": t_p[mask], "y_fit": y_p[mask],
                    "left_bound_ns": float(t_p[mask][0]), "weighting": weighting,
                    "lifetime_shift_ns": self.lifetime_shift_ns, "irf_t_ns": t_p,
                    "period": self.rep_time * 1e9 if decay_correction else None,
                    "exclude_seam": bool(exclude_seam), **self._seam_report(t_ns, seam),
                    **self._channel_report()})
        if isinstance(irf, PTU):
            out["irf_source"] = str(irf.filepath)
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
            times = (np.asarray(self.marker[marker_channel], dtype=float)
                     * (self._marker_unit or self.rep_time))
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
            return np.array(self.marker[marker_channel]) * (self._marker_unit or self.rep_time)
        warnings.warn(
            f"{self.filepath.stem}: marker channel {marker_channel} not found.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
