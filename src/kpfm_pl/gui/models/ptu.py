"""PL lifetime workflow: time trace, ROIs, lifetime fits, lifetime trace (no Qt).

``PTU`` keeps the most recent time trace internally and several methods
(``threshold``, ``lifetime_histogram``, ``laser_triggers``) work on that
cached trace. Other calls (ROI matrices, FLIDs, lifetime traces) silently
replace it. To keep results consistent, every function here re-builds the
time trace with the GUI's current :class:`TraceParams` before calling a
cache-dependent method, all under the file's lock.

A region of interest (ROI) is a box ``{"x_min", "x_max", "y_min", "y_max"}``
drawn on the time trace (time in s, counts per time bin). Its count range
only has a meaning at the bin time of that trace, so everything computed for
an ROI (histograms, ROI lifetime traces, FLIDs) uses the trace's bin time.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...core import Signal
from ...utils import TriggerPoint, trigger_points
from ..export import ExportPayload, build_metadata, describe_meta
from .files import LoadedFile

__all__ = [
    "TraceParams",
    "StateParams",
    "FitParams",
    "FitOptions",
    "LifetimeTraceParams",
    "TraceResult",
    "FitResult",
    "apply_lifetime_shift",
    "trigger_times",
    "trigger_options",
    "resolve_window",
    "roi_mask",
    "clip_roi",
    "compute_trace",
    "compute_states",
    "compute_histogram",
    "preview_histogram",
    "compute_fit",
    "compute_roi_fits",
    "compute_lifetime_trace",
    "fit_summary",
    "format_uncertainty",
    "tau_int_text",
    "shifted_histogram_x",
    "trace_payload",
    "histogram_payload",
    "fit_payload",
    "fits_payload",
    "lifetime_trace_payload",
]

#: Histogram sources selectable for the lifetime fit; "roi" also needs ``FitParams.roi``.
HISTOGRAM_SOURCES = (("all", "Whole time trace"), ("on", "On state"), ("off", "Off state"))
ROI_SOURCES = (("rois", "All ROIs combined"), ("each", "Each ROI separately"))
FIT_MODES = (("fraction", "Rising edge (fraction)"), ("inflection", "Inflection point"),
             ("manual", "Manual (start/end)"))
WEIGHTINGS = (("none", "None (least squares)"), ("poisson", "Poisson (1/√counts)"))
LOSSES = (("linear", "Linear (standard)"), ("soft_l1", "Soft L1"), ("huber", "Huber"),
          ("cauchy", "Cauchy"), ("arctan", "Arctan"))
#: Exponential components that can be fine-tuned in :class:`FitOptions`.
MAX_COMPONENTS = 4


@dataclass
class TraceParams:
    """Arguments of ``PTU.time_trace`` plus the marker channel and dark counts.

    ``start_trigger``/``end_trigger`` (1-based trigger numbers, see
    :func:`trigger_times`) replace ``tstart``/``tend`` when set.
    """

    bintime: float = 0.05
    tstart: float = 0.0
    tend: float | None = None
    laser_pulse: bool = True
    marker_channel: int = 4
    dark_counts: float = 0.0
    start_trigger: int | None = None
    end_trigger: int | None = None


@dataclass
class StateParams:
    """On/off thresholds as fractions of the time-trace maximum (``PTU.threshold``)."""

    off: float = 0.0
    on: float = 1.0


@dataclass
class FitParams:
    """Arguments of ``PTU.fit_lifetime`` and the histogram it is applied to.

    ``source`` is ``"all"``, ``"on"``, ``"off"``, ``"roi"`` (the ROI with the
    0-based index ``roi``), ``"rois"`` (all ROIs combined) or ``"each"`` (one
    fit per ROI, see :func:`compute_roi_fits`).
    """

    source: str = "all"
    n_exp: int = 2
    mode: str = "fraction"
    fraction: float = 0.95
    smoothing: float = 2.0
    start: float = 0.0
    end: float | None = None
    normalize: bool = False
    roi: int = 0

    def window_kwargs(self) -> dict[str, Any]:
        return {"mode": self.mode, "fraction": self.fraction, "smoothing": self.smoothing,
                "start": self.start, "end": self.end}


def _components(value: float | None = None) -> list[float | None]:
    return [value] * MAX_COMPONENTS


@dataclass
class FitOptions:
    """Advanced fit settings shared by the lifetime fit, lifetime trace and FLIDs.

    Lists hold one entry per exponential component (index 0 is component 1);
    ``None`` means automatic: ``MultiExpFit``'s own initial guess and its
    default bounds (``tau > 0``, ``A >= 0``, no upper limit). Lifetimes are in
    ns. Amplitudes are fractions of the histogram peak, so the same numbers
    work for raw-count fits and for the normalised per-bin fits. Entries
    beyond the fit's ``n_exp`` are ignored.

    ``weighting`` and ``loss`` select Poisson weighting and a robust loss
    (``scipy.optimize.least_squares``). ``f_scale`` is the residual size where
    a robust loss starts to down-weight outliers: in Poisson standard
    deviations with Poisson weighting, otherwise as a fraction of the peak.
    """

    tau_guess: list[float | None] = field(default_factory=_components)
    tau_min: list[float | None] = field(default_factory=_components)
    tau_max: list[float | None] = field(default_factory=_components)
    amp_guess: list[float | None] = field(default_factory=_components)
    amp_min: list[float | None] = field(default_factory=_components)
    amp_max: list[float | None] = field(default_factory=_components)
    weighting: str = "none"
    loss: str = "linear"
    f_scale: float = 1.0

    def is_default(self) -> bool:
        return not self.fit_kwargs(MAX_COMPONENTS)

    def validate(self) -> None:
        """Raise ``ValueError`` for empty bounds or guesses outside their bounds."""
        for name, guess, lo, hi in (("τ", self.tau_guess, self.tau_min, self.tau_max),
                                    ("A", self.amp_guess, self.amp_min, self.amp_max)):
            for i in range(MAX_COMPONENTS):
                g, a, b = guess[i], lo[i], hi[i]
                if a is not None and b is not None and a >= b:
                    raise ValueError(f"{name}{i + 1}: the minimum must be smaller than "
                                     "the maximum.")
                if g is not None and ((a is not None and g < a) or (b is not None and g > b)):
                    raise ValueError(f"{name}{i + 1}: the initial guess lies outside its "
                                     "bounds.")
        if self.f_scale <= 0:
            raise ValueError("The robust-loss scale must be positive.")

    def fit_kwargs(self, n_exp: int, amplitude_scale: float = 1.0) -> dict[str, Any]:
        """Keyword arguments for ``fit_lifetime``/``lifetime_trace``/``flid``.

        Only settings that differ from the defaults are returned, so default
        options reproduce the plain backend call. ``amplitude_scale`` is the
        histogram peak in the units of the fitted data (1 for normalised fits).
        """
        n = int(n_exp)
        scale = float(amplitude_scale)

        def scaled(values, factor, default):
            return [default if v is None else float(v) * factor for v in values[:n]]

        kwargs: dict[str, Any] = {}
        if any(v is not None for v in (*self.amp_min[:n], *self.amp_max[:n],
                                       *self.tau_min[:n], *self.tau_max[:n])):
            lower = scaled(self.amp_min, scale, 0.0) + scaled(self.tau_min, 1.0, 1e-10)
            upper = scaled(self.amp_max, scale, np.inf) + scaled(self.tau_max, 1.0, np.inf)
            kwargs["bounds"] = (np.array(lower), np.array(upper))
        if any(v is not None for v in (*self.amp_guess[:n], *self.tau_guess[:n])):
            # NaN entries are filled with MultiExpFit's automatic guess.
            kwargs["p0"] = np.array(scaled(self.amp_guess, scale, np.nan)
                                    + scaled(self.tau_guess, 1.0, np.nan))
        if self.weighting != "none":
            kwargs["weighting"] = self.weighting
        if self.loss != "linear":
            kwargs["loss"] = self.loss
            kwargs["f_scale"] = float(self.f_scale) * (1.0 if self.weighting == "poisson"
                                                       else scale)
        return kwargs


@dataclass
class LifetimeTraceParams:
    """Arguments of ``PTU.lifetime_trace`` not shared with the fit window.

    ``roi`` (0-based) restricts the trace to one ROI; it then uses the bin
    time of the time trace instead of ``bintime``.
    """

    bintime: float = 0.5
    n_exp: int = 1
    bins: int = 1
    roi: int | None = None


@dataclass
class TraceResult:
    """A time trace and its triggers.

    ``triggers`` and ``marker_times`` lie inside the trace's window;
    ``trigger_times`` are all trigger times of the file (for choosing a
    window) and ``window`` is the resolved ``(tstart, tend)``.
    """

    signal: Signal
    triggers: list[TriggerPoint] | None
    marker_times: np.ndarray | None
    params: TraceParams
    source: str = ""
    trigger_times: np.ndarray | None = None
    window: tuple[float, float | None] = (0.0, None)
    lifetime_shift_ns: float = 0.0


@dataclass
class FitResult:
    histogram: Signal
    fit: dict[str, Any]
    params: FitParams
    states: StateParams
    trace_params: TraceParams
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    roi_index: int | None = None
    rois: list[dict[str, float]] = field(default_factory=list)
    options: FitOptions = field(default_factory=FitOptions)


# ------------------------------------------------------------- lifetime shift
def apply_lifetime_shift(handle: LoadedFile, shift: float | str | None) -> float:
    """Set the circular lifetime shift of the file (``PTU.set_lifetime_shift``).

    ``None`` restores the raw data, a number shifts by that many ns,
    ``"auto"`` puts the histogram peak 5 ns after zero. Nothing is recomputed
    when the shift is unchanged. Returns the applied shift in ns.
    """
    with handle.lock:
        ptu = handle.reader
        if shift != ptu.lifetime_shift:
            ptu.set_lifetime_shift(shift)
        return ptu.lifetime_shift_ns


# ------------------------------------------------------------------ time window
def trigger_times(handle: LoadedFile, params: TraceParams) -> np.ndarray | None:
    """All laser-trigger times of the file in s (``None`` without triggers).

    Hardware markers when the file has them; for legacy files
    (``laser_pulse=False``) the two edges of the whole, centred trace. Caller
    must hold ``handle.lock`` (the legacy case replaces the cached trace).
    """
    ptu = handle.reader
    if ptu.marker.get(params.marker_channel):
        markers = ptu.get_markers(params.marker_channel)
        return None if markers is None else np.asarray(markers, dtype=float)
    if not params.laser_pulse:
        edges = ptu.time_trace(bintime=params.bintime, laser_pulse=False).meta.get("triggers")
        if edges:
            return np.array([p["time"] for p in edges], dtype=float)
    return None


def trigger_options(times: np.ndarray | None) -> list[tuple[int, str]]:
    """Drop-down entries ``(number, "#n – 12.345 s")`` for 1-based trigger numbers."""
    if times is None:
        return []
    return [(i, f"#{i} – {t:.3f} s") for i, t in enumerate(np.asarray(times, float), 1)]


def resolve_window(params: TraceParams,
                   times: np.ndarray | None) -> tuple[float, float | None]:
    """``(tstart, tend)`` of the trace, with trigger numbers replaced by their times."""
    n = 0 if times is None else len(times)

    def pick(number: int, name: str) -> float:
        if not 1 <= number <= n:
            raise ValueError(f"The time window {name} refers to trigger {number}, but the "
                             f"file has {n} trigger(s).")
        return float(times[number - 1])

    tstart = params.tstart if params.start_trigger is None else pick(params.start_trigger,
                                                                     "start")
    tend = params.tend if params.end_trigger is None else pick(params.end_trigger, "end")
    if tend is not None and tend <= tstart:
        raise ValueError(f"The time window ends ({tend:.3f} s) before it starts "
                         f"({tstart:.3f} s).")
    return float(tstart), tend


def _uses_triggers(params: TraceParams) -> bool:
    return params.start_trigger is not None or params.end_trigger is not None


def _window(handle: LoadedFile, params: TraceParams) -> tuple[float, float | None]:
    """Resolved time window; caller must hold ``handle.lock``."""
    return resolve_window(params, trigger_times(handle, params)
                          if _uses_triggers(params) else None)


def _build_trace(handle: LoadedFile, params: TraceParams,
                 times: np.ndarray | None = None) -> Signal:
    """(Re)build the cached time trace; caller must hold ``handle.lock``."""
    ptu = handle.reader
    tstart, tend = (resolve_window(params, times) if times is not None
                    else _window(handle, params))
    ptu.dark_counts = float(params.dark_counts)
    return ptu.time_trace(bintime=params.bintime, tstart=tstart, tend=tend,
                          laser_pulse=params.laser_pulse)


def _triggers_in(times: np.ndarray, signal: Signal) -> list[TriggerPoint] | None:
    """``trigger_points`` of the ``times`` inside ``signal`` (no warning for the rest)."""
    x = np.asarray(signal.x, dtype=float)
    if not x.size:
        return None
    half = 0.5 * float(np.median(np.diff(x))) if x.size > 1 else 0.0
    inside = times[(times >= x.min() - half) & (times <= x.max() + half)]
    return trigger_points(inside, x) or None if inside.size else None


def compute_trace(handle: LoadedFile, params: TraceParams) -> TraceResult:
    """Time trace plus laser triggers.

    ``marker_times`` are the exact hardware marker times inside the trace when
    the file has a marker channel, otherwise the trigger times from
    ``laser_triggers`` (the two edges of a legacy ``laser_pulse=False``
    trace) or ``None``.
    """
    ptu = handle.reader
    with handle.lock:
        times = trigger_times(handle, params)
        window = resolve_window(params, times)
        signal = _build_trace(handle, params, times)
        if ptu.marker.get(params.marker_channel):
            triggers = _triggers_in(times, signal) if times is not None else None
            if triggers:
                lo = float(signal.x[0])
                hi = float(signal.x[-1]) + params.bintime
                marker_times = times[(times >= lo) & (times < hi)]
            else:
                marker_times = None
        else:
            triggers = ptu.laser_triggers(params.marker_channel, reference=signal)
            marker_times = (np.array([p["time"] for p in triggers], dtype=float)
                            if triggers else None)
    return TraceResult(signal, triggers, marker_times, params, str(handle.path), times, window,
                       ptu.lifetime_shift_ns)


# ----------------------------------------------------------------------- ROIs
def roi_mask(signal: Signal, rois: Sequence[dict[str, float]]) -> np.ndarray:
    """Time bins of ``signal`` inside any of ``rois``.

    Same rule as ``PTU.lifetime_trace``/``PTU.flid``: a bin belongs to an ROI
    when it overlaps the ROI's time window ``[x_min, x_max)`` and its counts
    lie in ``[y_min, y_max)``.
    """
    x = np.asarray(signal.x, dtype=float)
    y = np.asarray(signal.y, dtype=float)
    durations = np.asarray(signal.meta.get("bin_durations",
                                           np.full(x.shape, signal.meta.get("bintime", 0.0))),
                           dtype=float)
    mask = np.zeros(x.shape, dtype=bool)
    for roi in rois:
        mask |= ((x + durations > roi["x_min"]) & (x < roi["x_max"])
                 & (y >= roi["y_min"]) & (y < roi["y_max"]))
    return mask


def clip_roi(roi: dict[str, float], trace: TraceResult) -> dict[str, float]:
    """``roi`` with its time window limited to the time window of ``trace``."""
    s = trace.signal
    if not len(s):
        return dict(roi)
    lo = float(s.x[0])
    hi = float(s.x[-1] + trace.params.bintime)
    return {**roi, "x_min": max(float(roi["x_min"]), lo), "x_max": min(float(roi["x_max"]), hi)}


def _roi_label(indices: Sequence[int]) -> str:
    if len(indices) == 1:
        return f"ROI {indices[0] + 1}"
    return "ROIs " + "+".join(str(i + 1) for i in indices)


def _roi_histogram(handle: LoadedFile, signal: Signal, params: TraceParams,
                   rois: Sequence[dict[str, float]], indices: Sequence[int]) -> Signal:
    """Lifetime histogram of the time bins inside ``rois[indices]``."""
    if not indices:
        raise ValueError("No ROI drawn: draw a box on the time trace first.")
    for i in indices:
        if not 0 <= i < len(rois):
            raise ValueError(f"ROI {i + 1} does not exist ({len(rois)} ROI(s) drawn).")
    selected = [rois[i] for i in indices]
    mask = roi_mask(signal, selected)
    matrix = np.asarray(signal.meta["matrix"])[mask]
    hist = handle.reader.lifetime_histogram(matrix, bintime=params.bintime)
    stem = signal.label.replace(" time trace", "")
    label = _roi_label(indices)
    return Signal(hist.x, hist.y, label=f"{label} ({stem})", units=dict(hist.units),
                  meta={**hist.meta, "rois": selected, "roi_numbers": [i + 1 for i in indices]})


# -------------------------------------------------------------------- compute
def compute_states(handle: LoadedFile, params: TraceParams,
                   states: StateParams) -> dict[str, Signal]:
    """``{"on": Signal, "off": Signal}`` lifetime histograms of the two states."""
    with handle.lock:
        _build_trace(handle, params)
        return handle.reader.threshold(off=states.off, on=states.on)


def compute_histogram(handle: LoadedFile, params: TraceParams, fit: FitParams,
                      states: StateParams,
                      rois: Sequence[dict[str, float]] = ()) -> Signal:
    """The lifetime histogram selected by ``fit.source`` (not ``"each"``)."""
    with handle.lock:
        signal = _build_trace(handle, params)
        if fit.source == "all":
            return handle.reader.lifetime_histogram()
        if fit.source in ("on", "off"):
            return handle.reader.threshold(off=states.off, on=states.on)[fit.source]
        if fit.source == "roi":
            return _roi_histogram(handle, signal, params, rois, [fit.roi])
        if fit.source == "rois":
            return _roi_histogram(handle, signal, params, rois, list(range(len(rois))))
    raise ValueError(f"Unknown histogram source {fit.source!r}.")


def preview_histogram(handle: LoadedFile, params: TraceParams, fit: FitParams,
                      states: StateParams,
                      rois: Sequence[dict[str, float]] = ()) -> Signal:
    """Histogram shown before fitting: ``fit.source`` if possible, else the whole trace.

    ``"each"`` previews all ROIs combined.
    """
    source = "rois" if fit.source == "each" else fit.source
    try:
        return compute_histogram(handle, params, dataclasses.replace(fit, source=source),
                                 states, rois)
    except ValueError:
        return compute_histogram(handle, params, dataclasses.replace(fit, source="all"),
                                 states, rois)


def _fit_histogram(handle: LoadedFile, histogram: Signal, fit: FitParams,
                   options: FitOptions) -> dict[str, Any]:
    y = np.asarray(histogram.y, dtype=float)
    if not np.any(y):
        name = histogram.label.split(" (")[0] if histogram.label else fit.source
        raise ValueError(f"The {name!r} histogram contains no photons; adjust the "
                         "on/off thresholds or the ROI.")
    scale = 1.0 if fit.normalize else float(np.max(y))
    return handle.reader.fit_lifetime(histogram, n_exp=fit.n_exp, normalize=fit.normalize,
                                      **fit.window_kwargs(),
                                      **options.fit_kwargs(fit.n_exp, scale))


def compute_fit(handle: LoadedFile, params: TraceParams, fit: FitParams,
                states: StateParams, rois: Sequence[dict[str, float]] = (),
                options: FitOptions | None = None) -> FitResult:
    """Fit the selected lifetime histogram with ``fit.n_exp`` exponentials."""
    options = options or FitOptions()
    options.validate()
    with handle.lock:
        histogram = compute_histogram(handle, params, fit, states, rois)
        result = _fit_histogram(handle, histogram, fit, options)
    return FitResult(histogram, result, fit, states, params, str(handle.path),
                     roi_index=fit.roi if fit.source == "roi" else None,
                     rois=[dict(r) for r in rois], options=options)


def compute_roi_fits(handle: LoadedFile, params: TraceParams, fit: FitParams,
                     states: StateParams, rois: Sequence[dict[str, float]],
                     options: FitOptions | None = None,
                     ) -> tuple[list[FitResult], list[tuple[int, str]]]:
    """One fit per ROI (``fit.source == "each"``).

    Returns the fits and the ``(index, reason)`` of ROIs that could not be
    fitted (e.g. no photons), like ``compute_flids``.
    """
    options = options or FitOptions()
    options.validate()
    if not rois:
        raise ValueError("No ROI drawn: draw a box on the time trace first.")
    results, skipped = [], []
    with handle.lock:
        signal = _build_trace(handle, params)
        for i in range(len(rois)):
            roi_fit = FitParams(**{**asdict(fit), "source": "roi", "roi": i})
            try:
                histogram = _roi_histogram(handle, signal, params, rois, [i])
                result = _fit_histogram(handle, histogram, roi_fit, options)
            except ValueError as exc:
                skipped.append((i, str(exc)))
                continue
            results.append(FitResult(histogram, result, roi_fit, states, params,
                                     str(handle.path), roi_index=i,
                                     rois=[dict(r) for r in rois], options=options))
    return results, skipped


def compute_lifetime_trace(handle: LoadedFile, params: TraceParams, fit: FitParams,
                           lt: LifetimeTraceParams,
                           progress: Callable[[int, int], Any] | None = None, *,
                           rois: Sequence[dict[str, float]] = (),
                           options: FitOptions | None = None) -> Signal:
    """Mean fitted lifetime per time bin within the trace's (or an ROI's) window."""
    options = options or FitOptions()
    options.validate()
    kwargs = {**fit.window_kwargs(), **options.fit_kwargs(lt.n_exp)}
    with handle.lock:
        ptu = handle.reader
        if lt.roi is None:
            tstart, tend = _window(handle, params)
            ptu.dark_counts = float(params.dark_counts)
            return ptu.lifetime_trace(x_min=tstart, x_max=tend, bintime=lt.bintime,
                                      n_exp=lt.n_exp, bins=lt.bins,
                                      laser_pulse=params.laser_pulse, progress=progress,
                                      **kwargs)
        if not 0 <= lt.roi < len(rois):
            raise ValueError(f"ROI {lt.roi + 1} does not exist ({len(rois)} ROI(s) drawn).")
        trace = compute_trace(handle, params)
        roi = clip_roi(rois[lt.roi], trace)
        signal = ptu.lifetime_trace(roi=roi, bintime=params.bintime, n_exp=lt.n_exp,
                                    bins=lt.bins, laser_pulse=params.laser_pulse,
                                    progress=progress, **kwargs)
    signal.label = f"{signal.label} ROI {lt.roi + 1}"
    return signal


# ------------------------------------------------------------------ summaries
def fit_summary(fit: dict[str, Any]) -> dict[str, Any]:
    """Scalar fit results: amplitudes, lifetimes (ns), offset, fit quality, window."""
    quality = fit.get("quality", {})
    summary: dict[str, Any] = {}
    for i, (a, tau) in enumerate(zip(np.atleast_1d(fit["A"]), np.atleast_1d(fit["tau"])), 1):
        summary[f"A{i}"] = float(a)
        summary[f"tau{i}_ns"] = float(tau)
    if "tau_int" in fit:
        summary["tau_int_ns"] = float(fit["tau_int"])
        summary["tau_int_err_ns"] = float(fit.get("tau_int_err", np.nan))
    summary["offset"] = float(fit.get("offset", 0.0))
    summary["reduced_chi_squared"] = float(quality.get("reduced_chi_squared", np.nan))
    summary["r_squared"] = float(quality.get("r_squared", np.nan))
    summary["success"] = bool(fit.get("success", False))
    summary["fit_start_ns"] = float(fit.get("left_bound_ns", np.nan))
    summary["fit_end_ns"] = float(np.max(fit["t_fit"])) if len(fit.get("t_fit", [])) else np.nan
    summary["fit_points"] = int(len(fit.get("t_fit", [])))
    if fit.get("lifetime_shift_ns"):
        summary["lifetime_shift_ns"] = float(fit["lifetime_shift_ns"])
    return summary


def format_uncertainty(value: float, error: float) -> str:
    """``"3.42 ± 0.05"``: the error to 1 significant digit (2 if it starts with 1 or 2).

    The value is rounded to the same decimal place; without a finite error
    it is shown with 4 significant digits.
    """
    if not np.isfinite(value):
        return "–"
    if not np.isfinite(error) or error <= 0:
        return f"{value:.4g}"
    exponent = int(np.floor(np.log10(error)))
    leading = error / 10.0 ** exponent
    digits = 2 if leading < 3 else 1
    decimals = digits - 1 - exponent
    if round(error, decimals) >= 10.0 ** (exponent + 1):  # e.g. 0.096 -> 0.10
        decimals -= 1
    shown = max(decimals, 0)
    return f"{round(value, decimals):.{shown}f} ± {round(error, decimals):.{shown}f}"


def tau_int_text(fit: dict[str, Any]) -> str | None:
    """``"τ_int = 3.42 ± 0.05 ns"``: intensity-weighted mean lifetime with its error.

    Equal to ``τ1`` (with its error) for one exponential.
    """
    if "tau_int" not in fit:
        return None
    return f"τ_int = {format_uncertainty(fit['tau_int'], fit.get('tau_int_err', np.nan))} ns"


def quality_text(fit: dict[str, Any]) -> str:
    """Short goodness-of-fit string for plot titles, e.g. ``"χ²ᵣ = 1.08, R² = 0.998"``."""
    quality = fit.get("quality", {})
    chi = quality.get("reduced_chi_squared", np.nan)
    r2 = quality.get("r_squared", np.nan)
    return f"χ²ᵣ = {chi:.3g}, R² = {r2:.4f}"


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
        results={"time_window_s": list(result.window),
                 "lifetime_shift_ns": result.lifetime_shift_ns,
                 "laser_trigger_times_s": triggers,
                 "marker_times_s": result.marker_times,
                 **describe_meta(s.meta)},
    )
    return ExportPayload(meta, [s], default_name=f"{s.label.replace(' ', '_')}")


def _fit_signals(result: FitResult, prefix: str = "") -> list[Signal]:
    fit = result.fit
    hist = result.histogram
    name = prefix or f"{result.params.source} histogram"
    data = Signal(fit["shifted_histogram_x"], hist.y, label=name,
                  units={"x": "ns", "y": hist.units["y"]})
    curve = Signal(fit["t_fit"], fit["curve"], label=f"{prefix} fit".strip(),
                   units={"x": "ns", "y": "normalised" if result.params.normalize
                          else "counts"})
    residuals = fit.get("quality", {}).get("weighted_residuals")
    signals = [data, curve]
    if residuals is not None:
        signals.append(Signal(fit["t_fit"], np.asarray(residuals, dtype=float),
                              label=f"{prefix} weighted residuals".strip(),
                              units={"x": "ns", "y": "sigma"}))
    return signals


def _fit_parameters(result: FitResult) -> dict[str, Any]:
    params = _params(("time_trace", result.trace_params), ("states", result.states),
                     ("fit", result.params), ("fit_options", result.options))
    if result.rois:
        params["rois"] = result.rois
    return params


_SHIFT_NOTE = "Histogram time axis is shifted so the histogram maximum is at 0 ns."
_FIT_NOTE = f"{_SHIFT_NOTE} Weighted residuals: (counts - fit) / sqrt(max(counts, 1))."


def _file_stem(source: str, fallback: str = "lifetime") -> str:
    """Full file name without suffix, spaces as ``_`` (``"Data1 500nW.ptu"`` -> ``"Data1_500nW"``)."""
    return Path(source).stem.replace(" ", "_") if source else fallback


def _histogram_name(histogram: Signal, source: str) -> str:
    """File-name stem of a lifetime histogram, e.g. ``"Data1_500nW_ROI2"`` or ``"…_on"``."""
    name = _file_stem(source)
    label = histogram.label or ""
    if "roi_numbers" in histogram.meta or label.startswith(("on (", "off (")):
        name += "_" + label.split(" (")[0].replace(" ", "")
    return name


def shifted_histogram_x(histogram: Signal) -> np.ndarray:
    """Histogram time axis in ns with the histogram maximum at 0 ns."""
    y = np.asarray(histogram.y, dtype=float)
    x = np.asarray(histogram.x, dtype=float)
    return (x - x[int(np.argmax(y))]) * 1e9 if y.size else x


def histogram_payload(histogram: Signal, source: str, params: TraceParams,
                      states: StateParams,
                      rois: Sequence[dict[str, float]] = ()) -> ExportPayload:
    """The lifetime histogram as shown before a fit (no fit results)."""
    data = Signal(shifted_histogram_x(histogram), histogram.y, label=histogram.label,
                  units={"x": "ns", "y": histogram.units["y"]})
    parameters = _params(("time_trace", params), ("states", states))
    if rois:
        parameters["rois"] = [dict(r) for r in rois]
    meta = build_metadata(
        "PTU lifetime histogram", sources=[source], parameters=parameters,
        results={"histogram": describe_meta(histogram.meta)}, note=_SHIFT_NOTE,
    )
    return ExportPayload(meta, [data],
                         default_name=f"{_histogram_name(histogram, source)}_lifetime")


def fit_payload(result: FitResult) -> ExportPayload:
    fit = result.fit
    hist = result.histogram
    meta = build_metadata(
        "PTU lifetime fit", sources=[result.source],
        parameters=_fit_parameters(result),
        results={**fit_summary(fit), "histogram": describe_meta(hist.meta)},
        note=_FIT_NOTE,
    )
    stem = _histogram_name(hist, result.source)
    return ExportPayload(meta, _fit_signals(result), default_name=f"{stem}_lifetime_fit")


def fits_payload(results: Sequence[FitResult]) -> ExportPayload:
    """One export for several fits (``"each"`` ROI); a single fit uses :func:`fit_payload`."""
    if len(results) == 1:
        return fit_payload(results[0])
    first = results[0]
    signals: list[Signal] = []
    summaries = {}
    for result in results:
        name = f"ROI {result.roi_index + 1}"
        signals += _fit_signals(result, name)
        summaries[name] = {**fit_summary(result.fit),
                           "histogram": describe_meta(result.histogram.meta)}
    meta = build_metadata("PTU lifetime fits per ROI", sources=[first.source],
                          parameters=_fit_parameters(first), results=summaries,
                          note=_FIT_NOTE)
    return ExportPayload(meta, signals,
                         default_name=f"{_file_stem(first.source)}_ROIs_lifetime_fits")


def lifetime_trace_payload(signal: Signal, source: str, params: TraceParams,
                           fit: FitParams, lt: LifetimeTraceParams,
                           options: FitOptions | None = None,
                           rois: Sequence[dict[str, float]] = ()) -> ExportPayload:
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
    parameters = _params(("time_trace", params), ("fit_window", fit), ("lifetime_trace", lt),
                         ("fit_options", options or FitOptions()))
    if lt.roi is not None and 0 <= lt.roi < len(rois):
        parameters["roi"] = dict(rois[lt.roi])
    meta = build_metadata(
        "PTU lifetime trace", sources=[source], parameters=parameters,
        note="Empty lifetime cells are time bins without photons, outside the ROI or "
             "with a failed fit.",
    )
    return ExportPayload(meta, columns, default_name=signal.label.replace(" ", "_"))
