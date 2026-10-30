"""FLID workflow: one FLID map per region drawn on the time trace (no Qt).

Mirrors ``examples/example_flid.py``; used by the *PL lifetime* tab. A region of interest (ROI) is a box
``{"x_min", "x_max", "y_min", "y_max"}`` on the time trace (time in s, counts
per time bin). Because the count range of a box depends on the bin time, the
FLIDs are computed with the bin time of the trace the boxes were drawn on,
unless the automatic bin time raises it for an ROI with too few photons (its
count range is then scaled along). All maps share one lifetime range and one
intensity range (scaled per map with its bin time), so their bins and
occurrence values can be compared directly.

Computing a FLID has two steps: the slow per-bin lifetimes
(``PTU.lifetime_trace`` and/or ``PTU.arrival_time_trace``) and the fast map
(``flid_map``). :func:`compute_flids` does both; :func:`remap_flids` only
redoes the maps, e.g. after the smoothing or an axis setting changed.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ...core import Image, Signal
from ...ptu import flid_map
from ..export import ExportPayload, build_metadata, describe_meta
from .files import LoadedFile
from .ptu import FitOptions, TraceParams, TraceResult, clip_roi, compute_trace, roi_mask

__all__ = [
    "FlidParams",
    "FlidEntry",
    "FlidResult",
    "default_roi",
    "window_roi",
    "scale_rois",
    "compute_flids",
    "remap_flids",
    "count_check_text",
    "flid_payload",
    "bin_fits_payload",
]

#: Per-bin lifetimes the maps can show: fitted, mean arrival time, or both side by side.
METHODS = (("fit", "Fit (per bin)"), ("mean_arrival", "Mean arrival time (no fit)"),
           ("both", "Both, side by side"))
METHOD_NAMES = {"fit": "Fit", "mean_arrival": "Mean arrival"}
#: :class:`FlidParams` fields that only change the maps, not the per-bin lifetimes.
MAP_FIELDS = ("lifetime_bins", "intensity_bins", "sigma", "lifetime_min", "lifetime_max",
              "intensity_max")


@dataclass
class FlidParams:
    """Arguments of ``PTU.flid`` as entered in the GUI.

    ``lifetime_max=None`` lets every map span its own data (maps are then not
    directly comparable). ``intensity_max=None`` uses ``(0, trace max + 1)``,
    shared by all maps; with the automatic bin time it is scaled per map by
    its bin time.

    ``region`` chooses what is mapped: ``"each"`` (one map per ROI, or the
    :func:`default_roi` fallback without ROIs), ``"window"`` (one map of the
    whole time window of the trace, all count levels) or ``"roi"`` (only the
    ROI with the 0-based index ``roi``).

    ``mode``/``fraction``/``smoothing``/``start``/``end`` are the FLID's own
    fit window (part of each decay that is fitted or averaged), ``method`` is
    ``"fit"``, ``"mean_arrival"`` or ``"both"``. ``min_counts`` (0 disables),
    ``coverage`` and ``auto_bintime`` are the photon-count check of
    ``PTU.flid``.
    """

    n_exp: int = 1
    bins: int = 1
    lifetime_bins: int = 100
    intensity_bins: int = 100
    sigma: float = 1.5
    lifetime_min: float = 0.0
    lifetime_max: float | None = 20.0
    intensity_max: float | None = None
    mode: str = "fraction"
    fraction: float = 0.95
    smoothing: float = 2.0
    start: float = 0.0
    end: float | None = None
    region: str = "each"
    roi: int = 0
    method: str = "fit"
    truncation_correction: bool = True
    min_counts: float = 100.0
    coverage: float = 0.9
    auto_bintime: bool = False

    def lifetime_range(self) -> tuple[float, float] | None:
        if self.lifetime_max is None:
            return None
        if self.lifetime_max <= self.lifetime_min:
            raise ValueError("The lifetime range maximum must be larger than its minimum.")
        return float(self.lifetime_min), float(self.lifetime_max)

    def intensity_range(self, trace: Signal) -> tuple[float, float]:
        top = float(np.max(trace.y)) + 1.0 if len(trace) else 1.0
        return 0.0, float(self.intensity_max) if self.intensity_max else top

    def window_kwargs(self) -> dict[str, Any]:
        return {"mode": self.mode, "fraction": self.fraction, "smoothing": self.smoothing,
                "start": self.start, "end": self.end}

    def methods(self) -> list[str]:
        """The per-bin lifetimes to compute, in display order."""
        if self.method == "both":
            return ["fit", "mean_arrival"]
        if self.method not in ("fit", "mean_arrival"):
            raise ValueError(f"Unknown FLID method {self.method!r}.")
        return [self.method]

    def same_lifetimes(self, other: FlidParams) -> bool:
        """``True`` when ``other`` only differs in map settings (:data:`MAP_FIELDS`)."""
        mine = {k: v for k, v in asdict(self).items() if k not in MAP_FIELDS}
        theirs = {k: v for k, v in asdict(other).items() if k not in MAP_FIELDS}
        return mine == theirs


@dataclass
class FlidEntry:
    """The FLID(s) of one ROI; ``index`` is the ROI's position in the input list.

    ``traces`` holds the per-bin lifetimes (``"fit"`` and/or
    ``"mean_arrival"``) and ``images`` the maps built from them. ``roi`` is the
    box at the trace's bin time; ``bintime`` is the bin time actually used.
    """

    index: int
    roi: dict[str, float]
    traces: dict[str, Signal]
    images: dict[str, Image] = field(default_factory=dict)

    @property
    def methods(self) -> list[str]:
        return list(self.traces)

    @property
    def image(self) -> Image:
        """The first map (the fitted one when both were computed)."""
        return next(iter(self.images.values()))

    @property
    def bintime(self) -> float:
        return float(next(iter(self.traces.values())).meta["bintime"])

    @property
    def count_check(self) -> dict[str, Any] | None:
        return next(iter(self.traces.values())).meta.get("count_check")

    @property
    def n_fitted(self) -> int:
        """Time bins with a lifetime in the first map."""
        values = np.asarray(next(iter(self.traces.values())).y, dtype=float)
        return int(np.count_nonzero(np.isfinite(values)))

    @property
    def n_bins(self) -> int:
        return int(len(next(iter(self.traces.values()))))


@dataclass
class FlidResult:
    entries: list[FlidEntry]
    skipped: list[tuple[int, dict[str, float], str]]
    trace: TraceResult
    params: FlidParams
    lifetime_range: tuple[float, float] | None
    intensity_range: tuple[float, float]
    source: str = ""
    fallback: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    options: FitOptions = field(default_factory=FitOptions)

    def entry_name(self, entry: FlidEntry) -> str:
        """Display name of ``entry``: its ROI, or the window it stands for."""
        if self.params.region == "window":
            return "Whole time window"
        if self.fallback:
            return "Fallback window"
        return f"ROI {entry.index + 1}"

    def entry_intensity_range(self, entry: FlidEntry) -> tuple[float, float]:
        """The shared counts range, scaled to the bin time of ``entry``."""
        factor = entry.bintime / self.trace.params.bintime
        return self.intensity_range[0] * factor, self.intensity_range[1] * factor


def window_roi(trace: TraceResult) -> dict[str, float]:
    """The whole time window of ``trace`` with all count levels, as an ROI."""
    s = trace.signal
    top = float(np.max(s.y)) + 1.0 if len(s) else 1.0
    x_min = float(s.x[0]) if len(s) else 0.0
    x_max = float(s.x[-1] + trace.params.bintime) if len(s) else 0.0
    return {"x_min": x_min, "x_max": x_max, "y_min": 0.0, "y_max": top}


def default_roi(trace: TraceResult) -> dict[str, float]:
    """Fallback ROI when none was drawn.

    The window between the first and last laser trigger inside the trace (as
    in the example script), or the whole trace without triggers; all count
    levels.
    """
    roi = window_roi(trace)
    markers = trace.marker_times
    if markers is not None and len(markers) >= 2 and markers[-1] > markers[0]:
        roi.update(x_min=float(markers[0]), x_max=float(markers[-1]))
    return clip_roi(roi, trace)


def scale_rois(rois: Sequence[dict[str, float]], factor: float) -> list[dict[str, float]]:
    """Scale the count range of ``rois`` by ``factor`` (e.g. after a bin-time change).

    Counts per bin grow linearly with the bin time, so a box drawn at 0.5 s
    keeps roughly the same meaning at 1 s when its count range is doubled.
    """
    return [{**roi, "y_min": roi["y_min"] * factor, "y_max": roi["y_max"] * factor}
            for roi in rois]


def count_check_text(check: dict[str, Any] | None) -> str | None:
    """Advice for a failed photon-count check (``None`` if passed or disabled)."""
    if not check or check.get("passed"):
        return None
    return (f"only {check['fraction']:.0%} of the time bins of {check['bintime']:g} s "
            f"reach {check['min_counts']:g} photons ({check['coverage']:.0%} required); "
            f"use a bin time of at least {check['suggested_bintime']:g} s or the "
            "automatic bin time.")


class _Progress:
    """Overall ``progress(done, total)`` in time bins over several backend calls.

    Each call starts with an estimate of its bins, replaced by the true number
    as soon as the backend reports it (it differs with the automatic bin time).
    """

    def __init__(self, progress: Callable[[int, int], Any] | None,
                 estimates: Sequence[int]) -> None:
        self.progress = progress
        self.done = [0] * len(estimates)
        self.totals = [max(int(n), 1) for n in estimates]

    def part(self, k: int) -> Callable[[int, int], None]:
        def report(done: int, total: int) -> None:
            self.done[k], self.totals[k] = int(done), max(int(total), 1)
            self.emit()
        return report

    def finish(self, k: int) -> None:
        self.done[k] = self.totals[k]
        self.emit()

    def emit(self) -> None:
        if self.progress is not None:
            self.progress(sum(self.done), sum(self.totals))


def _selected_bins(trace: TraceResult, roi: dict[str, float]) -> int:
    return int(np.count_nonzero(roi_mask(trace.signal, [roi])))


def _build_maps(entry: FlidEntry, result: FlidResult) -> None:
    params = result.params
    for method, signal in entry.traces.items():
        entry.images[method] = flid_map(
            signal, lifetime_bins=params.lifetime_bins, intensity_bins=params.intensity_bins,
            lifetime_range=result.lifetime_range,
            intensity_range=result.entry_intensity_range(entry), sigma=params.sigma)


def compute_flids(handle: LoadedFile, trace_params: TraceParams, params: FlidParams,
                  rois: Sequence[dict[str, float]],
                  progress: Callable[[int, int], Any] | None = None,
                  options: FitOptions | None = None) -> FlidResult:
    """Compute the FLIDs of ``params.region`` with shared histogram ranges.

    ``"each"``: one FLID per ROI; without ROIs, :func:`default_roi` is used
    (``result.fallback`` is then ``True``). ``"window"``: one FLID of the
    whole trace window. ``"roi"``: one FLID of ``rois[params.roi]``; entry
    indices are always positions in ``rois``. ROIs are limited to the time window of the trace. ROIs that
    contain no photons are reported in ``result.skipped`` instead of raising.
    ``progress(done, total)`` reports the overall progress in time bins across
    all ROIs and methods; an exception raised by it (e.g. a cancellation)
    aborts the computation. ``options`` are the advanced fit settings of the
    per-bin fits. A failed photon-count check is not warned about by the
    backend; :func:`count_check_text` of ``entry.count_check`` gives the
    advice per ROI.
    """
    lifetime_range = params.lifetime_range()
    methods = params.methods()
    options = options or FitOptions()
    options.validate()
    fit_kwargs = options.fit_kwargs(params.n_exp)
    with handle.lock:
        trace = compute_trace(handle, trace_params)
        intensity_range = params.intensity_range(trace.signal)
        fallback = False
        if params.region == "window":
            selected = [(0, window_roi(trace))]
        elif params.region == "roi":
            if not 0 <= params.roi < len(rois):
                raise ValueError(f"ROI {params.roi + 1} does not exist "
                                 f"({len(rois)} ROI(s) drawn).")
            selected = [(params.roi, clip_roi(dict(rois[params.roi]), trace))]
        elif params.region == "each":
            fallback = not rois
            selected = ([(0, default_roi(trace))] if fallback
                        else [(i, clip_roi(dict(r), trace)) for i, r in enumerate(rois)])
        else:
            raise ValueError(f"Unknown FLID region {params.region!r}.")

        ptu = handle.reader
        common = {"bintime": trace_params.bintime, "min_counts": params.min_counts or None,
                  "coverage": params.coverage, "auto_bintime": params.auto_bintime,
                  "laser_pulse": trace_params.laser_pulse, **params.window_kwargs()}
        tracker = _Progress(progress, [_selected_bins(trace, roi)
                                       for _, roi in selected for _ in methods])
        result = FlidResult([], [], trace, params, lifetime_range, intensity_range,
                            str(handle.path), fallback, options=options)
        for k, (i, roi) in enumerate(selected):
            parts = range(k * len(methods), (k + 1) * len(methods))
            traces: dict[str, Signal] = {}
            try:
                # The tab reports the count check per ROI (count_check_text).
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r".* photons \(",
                                            category=RuntimeWarning)
                    for part, method in zip(parts, methods):
                        if method == "fit":
                            traces[method] = ptu.lifetime_trace(
                                roi=roi, n_exp=params.n_exp, bins=params.bins,
                                progress=tracker.part(part), **common, **fit_kwargs)
                        else:
                            traces[method] = ptu.arrival_time_trace(
                                roi=roi, truncation_correction=params.truncation_correction,
                                progress=tracker.part(part), **common)
                        tracker.finish(part)
            except ValueError as exc:  # e.g. a box containing no photons
                result.skipped.append((i, roi, str(exc)))
                for part in parts:
                    tracker.finish(part)
                continue
            entry = FlidEntry(i, roi, traces)
            _build_maps(entry, result)
            result.entries.append(entry)
        tracker.emit()
    return result


def remap_flids(result: FlidResult, params: FlidParams) -> FlidResult:
    """``result`` with its maps rebuilt for the map settings of ``params``.

    Only :data:`MAP_FIELDS` may differ from ``result.params``; the per-bin
    lifetimes are reused, so nothing is fitted again.
    """
    if not params.same_lifetimes(result.params):
        raise ValueError("Only the map settings may change without recomputing the FLIDs.")
    lifetime_range = params.lifetime_range()
    new = dataclasses.replace(
        result, params=params, lifetime_range=lifetime_range,
        intensity_range=params.intensity_range(result.trace.signal),
        entries=[FlidEntry(e.index, e.roi, e.traces) for e in result.entries])
    for entry in new.entries:
        _build_maps(entry, new)
    return new


# ------------------------------------------------------------------ export
def _metadata(result: FlidResult, entry: FlidEntry, analysis: str) -> dict[str, Any]:
    image = entry.image
    return build_metadata(
        analysis, sources=[result.source],
        parameters={"time_trace": asdict(result.trace.params),
                    "flid": asdict(result.params),
                    "fit_options": asdict(result.options)},
        results={"region": result.entry_name(entry),
                 "roi_number": (None if result.params.region == "window"
                                else entry.index + 1),
                 "roi": entry.roi,
                 "roi_is_fallback": result.fallback,
                 "bintime_s": entry.bintime,
                 "count_check": entry.count_check,
                 "time_bins_in_roi": entry.n_bins,
                 "time_bins_fitted": entry.n_fitted,
                 "lifetime_range_ns": image.meta.get("lifetime_range"),
                 "intensity_range_counts": image.meta.get("intensity_range")},
    )


def _stem(result: FlidResult) -> str:
    return result.trace.signal.label.replace(" time trace", "").replace(" ", "_")


def flid_payload(result: FlidResult, entry: FlidEntry,
                 method: str | None = None) -> ExportPayload:
    """The FLID map of ``entry`` (``method``: its first map by default)."""
    method = method or entry.methods[0]
    meta = _metadata(result, entry, "FLID map")
    meta["results"]["method"] = method
    meta["note"] = ("Rows: counts per time bin (bin centres); columns: "
                    + ("mean-arrival lifetime" if method == "mean_arrival" else "fitted lifetime")
                    + " in ns (bin centres); values: occurrence (smoothed).")
    suffix = "_mean_arrival" if method == "mean_arrival" else ""
    return ExportPayload(meta, image=entry.images[method],
                         default_name=f"{_stem(result)}_FLID_ROI{entry.index + 1}{suffix}")


def bin_fits_payload(result: FlidResult, entry: FlidEntry) -> ExportPayload:
    """Per-time-bin results behind the FLID map(s) of ``entry``."""
    first = next(iter(entry.traces.values()))
    time = np.asarray(first.x, dtype=float)
    columns = [Signal(time, np.asarray(first.meta["counts"], dtype=float), label="counts",
                      units={"x": "s", "y": "counts per bin"})]
    if "fit" in entry.traces:
        m = entry.traces["fit"].meta
        taus = np.asarray(m["taus"], dtype=float)
        amps = np.asarray(m["amplitudes"], dtype=float)
        if taus.shape[1] > 1:
            columns.append(Signal(time, np.asarray(entry.traces["fit"].y, dtype=float),
                                  label="tau_int", units={"x": "s", "y": "ns"}))
        for i in range(taus.shape[1]):
            columns.append(Signal(time, taus[:, i], label=f"tau{i + 1}", units={"x": "s", "y": "ns"}))
            columns.append(Signal(time, amps[:, i], label=f"A{i + 1}", units={"x": "s", "y": ""}))
    if "mean_arrival" in entry.traces:
        arrival = entry.traces["mean_arrival"]
        columns.append(Signal(time, np.asarray(arrival.y, dtype=float),
                              label="tau_mean_arrival", units={"x": "s", "y": "ns"}))
        columns.append(Signal(time, np.asarray(arrival.meta["mean_arrival"], dtype=float),
                              label="mean_arrival_uncorrected", units={"x": "s", "y": "ns"}))
    meta = _metadata(result, entry, "FLID per-bin fits")
    meta["image_meta"] = describe_meta(entry.image.meta)
    meta["note"] = "Empty cells: time bins outside the ROI count range, empty, or failed fits."
    return ExportPayload(meta, columns, default_name=f"{_stem(result)}_FLID_ROI{entry.index + 1}_bins")
