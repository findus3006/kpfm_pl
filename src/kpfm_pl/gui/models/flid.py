"""FLID workflow: one FLID map per region drawn on the time trace (no Qt).

Mirrors ``examples/example_flid.py``; used by the *PL lifetime* tab. A region of interest (ROI) is a box
``{"x_min", "x_max", "y_min", "y_max"}`` on the time trace (time in s, counts
per time bin). Because the count range of a box depends on the bin time, the
FLIDs are always computed with the bin time of the trace the boxes were drawn
on. All maps share one lifetime and one intensity range, so their bins and
occurrence values can be compared directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ...core import Image, Signal
from ..export import ExportPayload, build_metadata, describe_meta
from .files import LoadedFile
from .ptu import FitOptions, TraceParams, TraceResult, clip_roi, compute_trace

__all__ = [
    "FlidParams",
    "FlidEntry",
    "FlidResult",
    "default_roi",
    "window_roi",
    "scale_rois",
    "compute_flids",
    "flid_payload",
    "bin_fits_payload",
]

#: What the CSV export of one ROI contains.
EXPORT_CONTENTS = (("map", "FLID map (2-D histogram)"),
                   ("bins", "Per-bin fits (time, counts, lifetimes)"))


@dataclass
class FlidParams:
    """Arguments of ``PTU.flid`` as entered in the GUI.

    ``lifetime_max=None`` lets every map span its own data (maps are then not
    directly comparable). ``intensity_max=None`` uses ``(0, trace max + 1)``,
    shared by all maps.

    ``region`` chooses what is mapped: ``"each"`` (one map per ROI, or the
    :func:`default_roi` fallback without ROIs), ``"window"`` (one map of the
    whole time window of the trace, all count levels) or ``"roi"`` (only the
    ROI with the 0-based index ``roi``).
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


@dataclass
class FlidEntry:
    """The FLID of one ROI; ``index`` is the ROI's position in the input list."""

    index: int
    roi: dict[str, float]
    image: Image

    @property
    def n_fitted(self) -> int:
        taus = np.asarray(self.image.meta.get("taus", np.empty((0, 1))))
        return int(np.count_nonzero(np.all(np.isfinite(taus), axis=1))) if taus.size else 0

    @property
    def n_bins(self) -> int:
        return int(len(self.image.meta.get("time", ())))


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
    ``progress(done, total)`` reports the overall progress across all ROIs;
    an exception raised by it (e.g. a cancellation) aborts the computation.
    ``options`` are the advanced fit settings of the per-bin fits.
    """
    lifetime_range = params.lifetime_range()
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

        # Per-ROI totals are only known once fitting starts, so each ROI gets
        # an equal share of a fixed overall scale.
        scale = 1000
        total = scale * len(selected)

        entries, skipped = [], []
        for k, (i, roi) in enumerate(selected):
            def _sub(done: int, n: int, _k: int = k) -> None:
                if progress is not None:
                    progress(_k * scale + int(scale * done / max(n, 1)), total)

            try:
                image = handle.reader.flid(
                    roi=roi, bintime=trace_params.bintime, n_exp=params.n_exp,
                    bins=params.bins, lifetime_bins=params.lifetime_bins,
                    intensity_bins=params.intensity_bins, lifetime_range=lifetime_range,
                    intensity_range=intensity_range, sigma=params.sigma,
                    progress=_sub, **params.window_kwargs(), **fit_kwargs,
                )
            except ValueError as exc:  # e.g. a box containing no photons
                skipped.append((i, roi, str(exc)))
                continue
            entries.append(FlidEntry(i, roi, image))
        if progress is not None:
            progress(total, total)
    return FlidResult(entries, skipped, trace, params, lifetime_range, intensity_range,
                      str(handle.path), fallback, options=options)


# ------------------------------------------------------------------ export
def _metadata(result: FlidResult, entry: FlidEntry, analysis: str) -> dict[str, Any]:
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
                 "time_bins_in_roi": entry.n_bins,
                 "time_bins_fitted": entry.n_fitted,
                 "lifetime_range_ns": entry.image.meta.get("lifetime_range"),
                 "intensity_range_counts": entry.image.meta.get("intensity_range")},
    )


def flid_payload(result: FlidResult, entry: FlidEntry) -> ExportPayload:
    """The FLID map of ``entry`` with its ROI and parameters."""
    meta = _metadata(result, entry, "FLID map")
    meta["note"] = ("Rows: counts per time bin (bin centres); columns: fitted lifetime "
                    "in ns (bin centres); values: occurrence (smoothed).")
    stem = result.trace.signal.label.replace(" time trace", "").replace(" ", "_")
    return ExportPayload(meta, image=entry.image, default_name=f"{stem}_FLID_ROI{entry.index + 1}")


def bin_fits_payload(result: FlidResult, entry: FlidEntry) -> ExportPayload:
    """Per-time-bin fit results behind the FLID map of ``entry``."""
    m = entry.image.meta
    time = np.asarray(m["time"], dtype=float)
    taus = np.asarray(m["taus"], dtype=float)
    amps = np.asarray(m["amplitudes"], dtype=float)
    columns = [Signal(time, np.asarray(m["counts"], dtype=float), label="counts",
                      units={"x": "s", "y": "counts per bin"})]
    for i in range(taus.shape[1]):
        columns.append(Signal(time, taus[:, i], label=f"tau{i + 1}", units={"x": "s", "y": "ns"}))
        columns.append(Signal(time, amps[:, i], label=f"A{i + 1}", units={"x": "s", "y": ""}))
    meta = _metadata(result, entry, "FLID per-bin fits")
    meta["image_meta"] = describe_meta(m)
    meta["note"] = "Empty cells: time bins outside the ROI count range, empty, or failed fits."
    stem = result.trace.signal.label.replace(" time trace", "").replace(" ", "_")
    return ExportPayload(meta, columns, default_name=f"{stem}_FLID_ROI{entry.index + 1}_bins")
