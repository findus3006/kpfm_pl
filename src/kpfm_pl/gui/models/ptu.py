"""PTU workflow: time trace, on/off states, lifetime fit and lifetime trace (no Qt).

``PTU`` keeps the most recent time trace internally and several methods
(``threshold``, ``lifetime_histogram``, ``laser_triggers``) work on that
cached trace. Other calls (ROI matrices, FLIDs, lifetime traces) silently
replace it. To keep results consistent, every function here re-builds the
time trace with the GUI's current :class:`TraceParams` before calling a
cache-dependent method, all under the file's lock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ...core import Signal
from ...utils import TriggerPoint
from ..export import ExportPayload, build_metadata, describe_meta
from .files import LoadedFile

__all__ = [
    "TraceParams",
    "StateParams",
    "FitParams",
    "LifetimeTraceParams",
    "TraceResult",
    "FitResult",
    "compute_trace",
    "compute_states",
    "compute_histogram",
    "compute_fit",
    "compute_lifetime_trace",
    "fit_summary",
    "trace_payload",
    "fit_payload",
    "lifetime_trace_payload",
]

#: Histogram sources selectable for the lifetime fit.
HISTOGRAM_SOURCES = (("all", "Whole time trace"), ("on", "On state"), ("off", "Off state"))
FIT_MODES = (("fraction", "Rising edge (fraction)"), ("inflection", "Inflection point"),
             ("manual", "Manual (start/end)"))


@dataclass
class TraceParams:
    """Arguments of ``PTU.time_trace`` plus the marker channel and dark counts."""

    bintime: float = 0.05
    tstart: float = 0.0
    tend: float | None = None
    laser_pulse: bool = True
    marker_channel: int = 4
    dark_counts: float = 0.0


@dataclass
class StateParams:
    """On/off thresholds as fractions of the time-trace maximum (``PTU.threshold``)."""

    off: float = 0.0
    on: float = 1.0


@dataclass
class FitParams:
    """Arguments of ``PTU.fit_lifetime`` and the histogram it is applied to."""

    source: str = "all"
    n_exp: int = 2
    mode: str = "fraction"
    fraction: float = 0.95
    smoothing: float = 2.0
    start: float = 0.0
    end: float | None = None
    normalize: bool = False

    def window_kwargs(self) -> dict[str, Any]:
        return {"mode": self.mode, "fraction": self.fraction, "smoothing": self.smoothing,
                "start": self.start, "end": self.end}


@dataclass
class LifetimeTraceParams:
    """Arguments of ``PTU.lifetime_trace`` not shared with the fit window."""

    bintime: float = 0.5
    n_exp: int = 1
    bins: int = 1


@dataclass
class TraceResult:
    signal: Signal
    triggers: list[TriggerPoint] | None
    marker_times: np.ndarray | None
    params: TraceParams
    source: str = ""


@dataclass
class FitResult:
    histogram: Signal
    fit: dict[str, Any]
    params: FitParams
    states: StateParams
    trace_params: TraceParams
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _build_trace(handle: LoadedFile, params: TraceParams) -> Signal:
    """(Re)build the cached time trace; caller must hold ``handle.lock``."""
    ptu = handle.reader
    ptu.dark_counts = float(params.dark_counts)
    return ptu.time_trace(bintime=params.bintime, tstart=params.tstart,
                          tend=params.tend, laser_pulse=params.laser_pulse)


def compute_trace(handle: LoadedFile, params: TraceParams) -> TraceResult:
    """Time trace plus laser triggers.

    ``marker_times`` are the exact hardware marker times when the file has a
    marker channel, otherwise the trigger times from ``laser_triggers``
    (the two edges of a legacy ``laser_pulse=False`` trace) or ``None``.
    """
    ptu = handle.reader
    with handle.lock:
        signal = _build_trace(handle, params)
        triggers = ptu.laser_triggers(params.marker_channel, reference=signal)
        if ptu.marker.get(params.marker_channel):
            marker_times = ptu.get_markers(params.marker_channel)
        elif triggers:
            marker_times = np.array([p["time"] for p in triggers], dtype=float)
        else:
            marker_times = None
    return TraceResult(signal, triggers, marker_times, params, str(handle.path))


def compute_states(handle: LoadedFile, params: TraceParams,
                   states: StateParams) -> dict[str, Signal]:
    """``{"on": Signal, "off": Signal}`` lifetime histograms of the two states."""
    with handle.lock:
        _build_trace(handle, params)
        return handle.reader.threshold(off=states.off, on=states.on)


def compute_histogram(handle: LoadedFile, params: TraceParams, fit: FitParams,
               states: StateParams) -> Signal:
    with handle.lock:
        _build_trace(handle, params)
        if fit.source == "all":
            return handle.reader.lifetime_histogram()
        return handle.reader.threshold(off=states.off, on=states.on)[fit.source]


def compute_fit(handle: LoadedFile, params: TraceParams, fit: FitParams,
                states: StateParams) -> FitResult:
    """Fit the selected lifetime histogram with ``fit.n_exp`` exponentials."""
    with handle.lock:
        histogram = compute_histogram(handle, params, fit, states)
        if not np.any(histogram.y):
            raise ValueError(f"The {fit.source!r} histogram contains no photons; "
                             "adjust the on/off thresholds.")
        result = handle.reader.fit_lifetime(histogram, n_exp=fit.n_exp,
                                            normalize=fit.normalize, **fit.window_kwargs())
    return FitResult(histogram, result, fit, states, params, str(handle.path))


def compute_lifetime_trace(handle: LoadedFile, params: TraceParams, fit: FitParams,
                           lt: LifetimeTraceParams,
                           progress: Callable[[int, int], Any] | None = None) -> Signal:
    """Mean fitted lifetime per time bin within the trace's time window."""
    with handle.lock:
        handle.reader.dark_counts = float(params.dark_counts)
        return handle.reader.lifetime_trace(
            x_min=params.tstart, x_max=params.tend, bintime=lt.bintime,
            n_exp=lt.n_exp, bins=lt.bins, laser_pulse=params.laser_pulse,
            progress=progress, **fit.window_kwargs(),
        )


# ------------------------------------------------------------------ summaries
def fit_summary(fit: dict[str, Any]) -> dict[str, Any]:
    """Scalar fit results: amplitudes, lifetimes (ns), offset, R², window start."""
    quality = fit.get("quality", {})
    summary: dict[str, Any] = {}
    for i, (a, tau) in enumerate(zip(np.atleast_1d(fit["A"]), np.atleast_1d(fit["tau"])), 1):
        summary[f"A{i}"] = float(a)
        summary[f"tau{i}_ns"] = float(tau)
    summary["offset"] = float(fit.get("offset", 0.0))
    summary["r_squared"] = float(quality.get("r_squared", np.nan))
    summary["success"] = bool(fit.get("success", False))
    summary["fit_start_ns"] = float(fit.get("left_bound_ns", np.nan))
    summary["fit_end_ns"] = float(np.max(fit["t_fit"])) if len(fit.get("t_fit", [])) else np.nan
    return summary


# ---------------------------------------------------------------- export
def _params(*objs) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, obj in objs:
        out[name] = asdict(obj)
    return out


def trace_payload(result: TraceResult) -> ExportPayload:
    s = result.signal
    triggers = [p["time"] for p in result.triggers] if result.triggers else []
    meta = build_metadata(
        "PTU time trace", sources=[result.source],
        parameters=_params(("time_trace", result.params)),
        results={"laser_trigger_times_s": triggers,
                 "marker_times_s": result.marker_times,
                 **describe_meta(s.meta)},
    )
    return ExportPayload(meta, [s], default_name=f"{s.label.replace(' ', '_')}")


def fit_payload(result: FitResult) -> ExportPayload:
    fit = result.fit
    hist = result.histogram
    data = Signal(fit["shifted_histogram_x"], hist.y, label=f"{result.params.source} histogram",
                  units={"x": "ns", "y": hist.units["y"]})
    curve = Signal(fit["t_fit"], fit["curve"], label="fit",
                   units={"x": "ns", "y": "normalised" if result.params.normalize else "counts"})
    meta = build_metadata(
        "PTU lifetime fit", sources=[result.source],
        parameters=_params(("time_trace", result.trace_params), ("states", result.states),
                           ("fit", result.params)),
        results={**fit_summary(fit), "histogram": describe_meta(hist.meta)},
        note="Histogram time axis is shifted so the histogram maximum is at 0 ns.",
    )
    stem = result.histogram.label.split(" ")[0] if result.histogram.label else "lifetime"
    return ExportPayload(meta, [data, curve], default_name=f"{stem}_lifetime_fit")


def lifetime_trace_payload(signal: Signal, source: str, params: TraceParams,
                           fit: FitParams, lt: LifetimeTraceParams) -> ExportPayload:
    taus = np.asarray(signal.meta.get("taus", np.empty((len(signal), 0))))
    amps = np.asarray(signal.meta.get("amplitudes", np.empty((len(signal), 0))))
    columns = [signal]
    counts = signal.meta.get("counts")
    if counts is not None and len(counts) == len(signal):
        columns.append(Signal(signal.x, np.asarray(counts, dtype=float), label="counts",
                              units={"x": "s", "y": "counts"}))
    for i in range(taus.shape[1] if taus.ndim == 2 else 0):
        columns.append(Signal(signal.x, taus[:, i], label=f"tau{i + 1}",
                              units={"x": "s", "y": "ns"}))
        columns.append(Signal(signal.x, amps[:, i], label=f"A{i + 1}",
                              units={"x": "s", "y": ""}))
    meta = build_metadata(
        "PTU lifetime trace", sources=[source],
        parameters=_params(("time_trace", params), ("fit_window", fit), ("lifetime_trace", lt)),
        note="Empty lifetime cells are time bins without photons or with a failed fit.",
    )
    return ExportPayload(meta, columns, default_name=signal.label.replace(" ", "_"))
