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
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...core import Signal
from ...fitting import MLEFallbackWarning
from ...utils import TriggerPoint, trigger_points
from ..export import ExportPayload, build_metadata, describe_meta
from .files import LoadedFile

__all__ = [
    "TraceParams",
    "StateParams",
    "FitParams",
    "FitOptions",
    "LifetimeTraceParams",
    "IRFParams",
    "load_irf",
    "irf_trace_kwargs",
    "background_trace_kwargs",
    "TraceResult",
    "FitResult",
    "apply_lifetime_shift",
    "apply_channels",
    "channel_options",
    "describe_file",
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
    "COMPONENT_COLUMNS",
    "ComponentRow",
    "component_columns",
    "component_rows",
    "quality_keys",
    "summary_text",
    "mle_fallback_text",
    "bin_fallback_text",
    "bin_runs",
    "BicComparison",
    "same_fit_data",
    "compare_bic",
    "format_uncertainty",
    "tau_int_text",
    "mean_text",
    "error_key",
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
FIT_MODES = (("fraction", "Falling edge (fraction)"), ("inflection", "Inflection point"),
             ("manual", "Manual (start/end)"))
WEIGHTINGS = (("mle", "Poisson MLE (maximum likelihood)"),
              ("poisson", "Poisson least squares (1/√counts)"),
              ("none", "None (plain least squares)"))
#: Backend defaults (``PTU.fit_lifetime`` / ``PTU.lifetime_trace`` and ``PTU.flid``).
FIT_DEFAULT_WEIGHTING = "mle"
BIN_DEFAULT_WEIGHTING = "poisson"
#: Defaults of the over-fitting warnings (``fitting.fit_warnings``).
DIAGNOSTIC_DEFAULTS = {"amp_sigma": 2.0, "min_fraction": 0.01, "merge_sigma": 2.0,
                       "merge_rel": 0.1, "bound_rel": 0.01, "period_fraction": 0.5}
#: Default |ΔBIC| below which two fits count as equally good (Kass & Raftery:
#: > 10 is "very strong" evidence).
BIC_THRESHOLD = 10.0
#: Mean lifetimes selectable in :class:`FitOptions` (``kpfm_pl.fitting.mean_lifetime``).
MEAN_LIFETIMES = (("intensity", "Intensity-weighted τ_int = Σaτ² / Σaτ"),
                  ("amplitude", "Amplitude-weighted τ_amp = Σaτ / Σa"))
#: Table/title symbol of each mean lifetime.
MEAN_SYMBOLS = {"intensity": "τ_int", "amplitude": "τ_amp"}
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

    ``prerise`` also fits the bins that lie more than ``prerise_gap`` ns
    before the histogram maximum (tail fits only; they hold the tail of the
    previous laser pulse). ``exclude_seam`` leaves out the lifetime bins that
    are not fully inside one laser period (``PTU.seam_bins``).
    """

    source: str = "all"
    n_exp: int = 2
    mode: str = "fraction"
    fraction: float = 0.95
    smoothing: float = 2.0
    start: float = 0.0
    end: float | None = None
    roi: int = 0
    prerise: bool = False
    prerise_gap: float = 2.0
    exclude_seam: bool = True

    def window_kwargs(self) -> dict[str, Any]:
        """Fit-window arguments; the pre-rise ones only when switched on.

        ``exclude_seam`` is the backend default and only passed when off.
        """
        kwargs = {"mode": self.mode, "fraction": self.fraction, "smoothing": self.smoothing,
                  "start": self.start, "end": self.end}
        if self.prerise:
            kwargs.update(prerise=True, prerise_gap=float(self.prerise_gap))
        if not self.exclude_seam:
            kwargs["exclude_seam"] = False
        return kwargs

    def window_values(self) -> dict[str, Any]:
        """All fit-window settings (for copying them to another form)."""
        return {"mode": self.mode, "fraction": self.fraction, "smoothing": self.smoothing,
                "start": self.start, "end": self.end, "prerise": self.prerise,
                "prerise_gap": self.prerise_gap, "exclude_seam": self.exclude_seam}


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

    ``weighting`` selects the fit statistic of the lifetime fit (Poisson
    maximum likelihood by default), ``bin_weighting`` that of the per-bin
    fits of the lifetime trace and the FLIDs (Poisson-weighted least squares
    by default, several times faster); both follow the backend defaults.
    ``loss`` selects a robust loss (``scipy.optimize.least_squares``). The
    maximum likelihood has no robust loss: with one, those fits use
    Poisson-weighted least squares (:meth:`effective_weighting`). ``f_scale``
    is the residual size where a robust loss starts to down-weight outliers:
    in Poisson standard deviations, or as a fraction of the peak without
    weighting.

    ``constrain_sum_A`` enforces ``sum(A) <= 1``. It is only applied to
    normalised fits (``amplitude_scale == 1``), where the amplitudes are
    fractions of the peak: the per-bin lifetime-trace/FLID fits. The lifetime
    fit works on raw counts and ignores it, because there the amplitudes are
    counts, not fractions.

    ``mean_lifetime`` (``"intensity"`` or ``"amplitude"``) selects the mean
    lifetime of the fit table and title, the lifetime trace and the FLID
    maps. ``decay_correction`` corrects the amplitudes for the incomplete
    decay of earlier laser pulses (all fits).

    ``bic_threshold`` is the |ΔBIC| from which a lifetime fit counts as better
    or worse than the previous one (see :func:`compare_bic`). ``amp_sigma``,
    ``min_fraction``, ``merge_sigma``, ``merge_rel`` and ``bound_rel`` are the
    thresholds of the over-fitting warnings of the lifetime fit
    (``kpfm_pl.fitting.fit_warnings``); ``period_fraction`` is the fraction of
    the laser period above which a lifetime is flagged when the background is
    fitted. These only affect the lifetime fit.
    """

    tau_guess: list[float | None] = field(default_factory=_components)
    tau_min: list[float | None] = field(default_factory=_components)
    tau_max: list[float | None] = field(default_factory=_components)
    amp_guess: list[float | None] = field(default_factory=_components)
    amp_min: list[float | None] = field(default_factory=_components)
    amp_max: list[float | None] = field(default_factory=_components)
    weighting: str = "mle"
    bin_weighting: str = "poisson"
    loss: str = "linear"
    f_scale: float = 1.0
    constrain_sum_A: bool = False
    mean_lifetime: str = "intensity"
    decay_correction: bool = True
    bic_threshold: float = BIC_THRESHOLD
    amp_sigma: float = DIAGNOSTIC_DEFAULTS["amp_sigma"]
    min_fraction: float = DIAGNOSTIC_DEFAULTS["min_fraction"]
    merge_sigma: float = DIAGNOSTIC_DEFAULTS["merge_sigma"]
    merge_rel: float = DIAGNOSTIC_DEFAULTS["merge_rel"]
    bound_rel: float = DIAGNOSTIC_DEFAULTS["bound_rel"]
    period_fraction: float = DIAGNOSTIC_DEFAULTS["period_fraction"]

    def is_default(self) -> bool:
        return (not self.fit_kwargs(MAX_COMPONENTS)
                and not self.fit_kwargs(MAX_COMPONENTS, per_bin=True))

    def chosen_weighting(self, per_bin: bool = False) -> str:
        """The selected weighting of the lifetime fit or (``per_bin``) the trace/FLID fits."""
        return self.bin_weighting if per_bin else self.weighting

    def effective_weighting(self, per_bin: bool = False) -> str:
        """The weighting the fits will use (``PTU.fit_lifetime``'s fallback).

        The Poisson MLE has no robust loss: combined with one, the backend
        fits by Poisson-weighted least squares and says so in the log.
        """
        chosen = self.chosen_weighting(per_bin)
        return "poisson" if chosen == "mle" and self.loss != "linear" else chosen

    def fallback_note(self, per_bin: bool = False) -> str | None:
        """Log message when the chosen weighting cannot be used (``None`` otherwise)."""
        if self.effective_weighting(per_bin) == self.chosen_weighting(per_bin):
            return None
        which = "lifetime trace / FLID fits" if per_bin else "lifetime fit"
        return (f"Poisson MLE has no robust loss: the {which} use(s) Poisson-weighted least "
                f"squares with the {self.loss} loss instead (Advanced fit settings).")

    def diagnostics(self) -> dict[str, float]:
        """Thresholds for ``PTU.fit_lifetime(diagnostics=...)``."""
        return {key: float(getattr(self, key)) for key in DIAGNOSTIC_DEFAULTS}

    def validate(self) -> None:
        """Raise ``ValueError`` for empty bounds or guesses outside their bounds."""
        for name, guess, lo, hi in (("τ", self.tau_guess, self.tau_min, self.tau_max),
                                    ("A", self.amp_guess, self.amp_min, self.amp_max)):
            for i in range(MAX_COMPONENTS):
                g, a, b = guess[i], lo[i], hi[i]
                if name == "A" and b is None and self.constrain_sum_A:
                    b = 1.0  # effective upper bound of the sum(A) constraint
                if a is not None and b is not None and a >= b:
                    raise ValueError(f"{name}{i + 1}: the minimum must be smaller than "
                                     "the maximum.")
                if g is not None and ((a is not None and g < a) or (b is not None and g > b)):
                    raise ValueError(f"{name}{i + 1}: the initial guess lies outside its "
                                     "bounds.")
        if self.f_scale <= 0:
            raise ValueError("The robust-loss scale must be positive.")
        if self.mean_lifetime not in MEAN_SYMBOLS:
            raise ValueError(f"Unknown mean lifetime {self.mean_lifetime!r}.")
        if self.bic_threshold < 0:
            raise ValueError("The BIC threshold must not be negative.")
        for key in DIAGNOSTIC_DEFAULTS:
            if getattr(self, key) < 0:
                raise ValueError(f"The warning threshold {key} must not be negative.")
        for name in ("weighting", "bin_weighting"):
            if getattr(self, name) not in dict(WEIGHTINGS):
                raise ValueError(f"Unknown {name} {getattr(self, name)!r}.")

    def fit_kwargs(self, n_exp: int, amplitude_scale: float = 1.0, *,
                   per_bin: bool = False) -> dict[str, Any]:
        """Keyword arguments for ``fit_lifetime`` or (``per_bin``) ``lifetime_trace``/``flid``.

        Only settings that differ from the defaults are returned, so default
        options reproduce the plain backend call. ``amplitude_scale`` is the
        histogram peak in the units of the fitted data (1 for normalised fits).
        ``per_bin`` selects :attr:`bin_weighting` instead of :attr:`weighting`.

        ``constrain_sum_A`` only enters the kwargs when ``amplitude_scale`` is
        1: on raw counts the amplitudes are not fractions of the peak, so the
        constraint would cap them at 1 count and ruin the fit.
        """
        n = int(n_exp)
        scale = float(amplitude_scale)
        constrained = bool(self.constrain_sum_A) and scale == 1.0

        def scaled(values, factor, default):
            return [default if v is None else float(v) * factor for v in values[:n]]

        kwargs: dict[str, Any] = {}
        if any(v is not None for v in (*self.amp_min[:n], *self.amp_max[:n],
                                       *self.tau_min[:n], *self.tau_max[:n])):
            lower = scaled(self.amp_min, scale, 0.0) + scaled(self.tau_min, 1.0, 1e-10)
            upper = scaled(self.amp_max, scale,
                           1.0 if constrained else np.inf) + scaled(self.tau_max, 1.0, np.inf)
            kwargs["bounds"] = (np.array(lower), np.array(upper))
        if any(v is not None for v in (*self.amp_guess[:n], *self.tau_guess[:n])):
            # NaN entries are filled with MultiExpFit's automatic guess.
            kwargs["p0"] = np.array(scaled(self.amp_guess, scale, np.nan)
                                    + scaled(self.tau_guess, 1.0, np.nan))
        # Passing the fallback explicitly keeps the backend from warning
        # again; the tab logs fallback_note() on every run instead.
        weighting = self.effective_weighting(per_bin)
        if weighting != (BIN_DEFAULT_WEIGHTING if per_bin else FIT_DEFAULT_WEIGHTING):
            kwargs["weighting"] = weighting
        if self.loss != "linear":
            kwargs["loss"] = self.loss
            kwargs["f_scale"] = float(self.f_scale) * (1.0 if weighting == "poisson"
                                                       else scale)
        if constrained:
            kwargs["constrain_sum_A"] = True
        if self.mean_lifetime != "intensity":
            kwargs["mean_lifetime"] = self.mean_lifetime
        if not self.decay_correction:
            kwargs["decay_correction"] = False
        return kwargs


@dataclass
class IRFParams:
    """Background and IRF settings of the lifetime fit (``PTU.fit_lifetime``).

    ``use`` switches the IRF reconvolution on (an IRF file must be loaded).
    ``fit_background`` frees the constant background, with or without the
    IRF; ``fit_scatter`` the scattered-light contribution (IRF only).
    """

    use: bool = False
    fit_background: bool = False
    fit_scatter: bool = False


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
    channels: tuple[int, ...] | None = None
    has_lifetime: bool = True


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
        if not ptu.has_lifetime:
            return 0.0                      # T2 file without laser pulses: nothing to shift
        if shift != ptu.lifetime_shift:
            ptu.set_lifetime_shift(shift)
        return ptu.lifetime_shift_ns


# ------------------------------------------------------------------ detectors
def apply_channels(handle: LoadedFile, channels: Sequence[int] | None) -> tuple[int, ...]:
    """Select the detector channel(s) of the file (``PTU.set_channels``).

    ``None`` uses the photons of all detectors together. Nothing is
    recomputed when the selection is unchanged. Returns the channels in use.
    """
    with handle.lock:
        ptu = handle.reader
        chosen = None if channels is None else tuple(sorted(int(c) for c in channels))
        if chosen != ptu.selected_channels:
            ptu.set_channels(chosen)
        return ptu.channels if chosen is None else chosen


def channel_options(handle: LoadedFile | None) -> list[tuple[Any, str]]:
    """Choices of the detector selector: all detectors, then each one."""
    channels = tuple(handle.reader.channels) if handle is not None else ()
    if len(channels) < 2:
        return [("all", "All detectors")]
    summed = " + ".join(str(c) for c in channels)
    return [("all", f"All detectors ({summed})")] + [(c, f"Detector {c}") for c in channels]


def describe_file(handle: LoadedFile) -> str:
    """One line on the record format of a PTU file and what it allows."""
    ptu = handle.reader
    parts = [f"{ptu.device}, {ptu.measurement_mode} mode"]
    if len(ptu.channels) > 1:
        parts.append("detectors " + ", ".join(str(c) for c in ptu.channels))
    if ptu.is_cw:
        parts.append("continuous wave")
    elif ptu.measurement_mode == "T2":
        if ptu.has_lifetime:
            divider = f", sync divider {ptu.sync_divider}" if ptu.sync_divider > 1 else ""
            parts.append(f"laser pulses on channel 0 (period {ptu.rep_time * 1e9:.5g} ns"
                         f"{divider}): lifetimes from the time since the last pulse, "
                         f"{ptu.resolution * 1e12:.4g} ps per bin")
        else:
            parts.append("no laser pulses recorded: time traces and triggers only, "
                         "no lifetimes")
    return "; ".join(parts) + "."


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
                       ptu.lifetime_shift_ns, ptu.selected_channels, ptu.has_lifetime)


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


def load_irf(path: str | Path) -> LoadedFile:
    """Read the IRF ``.ptu`` file (run it in a job; it can be large)."""
    from .files import load_ptu
    return load_ptu(path)


def _irf_kwargs(irf: LoadedFile | None, irf_params: IRFParams | None) -> dict[str, Any]:
    """``fit_lifetime`` arguments of the IRF and background settings."""
    if irf_params is None:
        return {}
    if irf is None or not irf_params.use:
        # Tail fit: only the fitted background applies.
        return {"fit_background": True} if irf_params.fit_background else {}
    return {"irf": irf.reader, "fit_background": irf_params.fit_background,
            "fit_scatter": irf_params.fit_scatter}


def _irf_extra(irf: LoadedFile | None, irf_params: IRFParams | None) -> dict[str, Any]:
    kwargs = _irf_kwargs(irf, irf_params)
    if "irf" in kwargs:
        return {"irf": {"source": str(irf.path), **asdict(irf_params)}}
    return {"fit_background": True} if kwargs else {}


def _fit_histogram(handle: LoadedFile, histogram: Signal, fit: FitParams,
                   options: FitOptions, irf: LoadedFile | None = None,
                   irf_params: IRFParams | None = None) -> dict[str, Any]:
    y = np.asarray(histogram.y, dtype=float)
    if not np.any(y):
        name = histogram.label.split(" (")[0] if histogram.label else fit.source
        raise ValueError(f"The {name!r} histogram contains no photons; adjust the "
                         "on/off thresholds or the ROI.")
    # Raw counts; amplitude options are fractions of the peak.
    scale = float(np.max(y))
    # The tab logs an MLE fallback from the result (mle_fallback_text) every
    # time; Python would show the backend's warning only once.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MLEFallbackWarning)
        return handle.reader.fit_lifetime(histogram, n_exp=fit.n_exp,
                                          diagnostics=options.diagnostics(),
                                          **fit.window_kwargs(),
                                          **options.fit_kwargs(fit.n_exp, scale),
                                          **_irf_kwargs(irf, irf_params))


def compute_fit(handle: LoadedFile, params: TraceParams, fit: FitParams,
                states: StateParams, rois: Sequence[dict[str, float]] = (),
                options: FitOptions | None = None, irf: LoadedFile | None = None,
                irf_params: IRFParams | None = None) -> FitResult:
    """Fit the selected lifetime histogram with ``fit.n_exp`` exponentials.

    With ``irf`` and ``irf_params.use`` the fit is an IRF reconvolution.
    """
    options = options or FitOptions()
    options.validate()
    with handle.lock:
        histogram = compute_histogram(handle, params, fit, states, rois)
        result = _fit_histogram(handle, histogram, fit, options, irf, irf_params)
    return FitResult(histogram, result, fit, states, params, str(handle.path),
                     extra=_irf_extra(irf, irf_params),
                     roi_index=fit.roi if fit.source == "roi" else None,
                     rois=[dict(r) for r in rois], options=options)


def compute_roi_fits(handle: LoadedFile, params: TraceParams, fit: FitParams,
                     states: StateParams, rois: Sequence[dict[str, float]],
                     options: FitOptions | None = None, irf: LoadedFile | None = None,
                     irf_params: IRFParams | None = None,
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
                result = _fit_histogram(handle, histogram, roi_fit, options, irf, irf_params)
            except ValueError as exc:
                skipped.append((i, str(exc)))
                continue
            results.append(FitResult(histogram, result, roi_fit, states, params,
                                     str(handle.path), extra=_irf_extra(irf, irf_params),
                                     roi_index=i, rois=[dict(r) for r in rois],
                                     options=options))
    return results, skipped


def compute_lifetime_trace(handle: LoadedFile, params: TraceParams, fit: FitParams,
                           lt: LifetimeTraceParams,
                           progress: Callable[[int, int], Any] | None = None, *,
                           rois: Sequence[dict[str, float]] = (),
                           options: FitOptions | None = None,
                           irf_kwargs: dict[str, Any] | None = None) -> Signal:
    """Mean fitted lifetime per time bin within the trace's (or an ROI's) window.

    ``irf_kwargs`` are extra per-bin arguments from the last lifetime fit:
    :func:`irf_trace_kwargs` switches the per-bin fits to IRF reconvolution,
    :func:`background_trace_kwargs` sets the background of the tail fits.
    """
    options = options or FitOptions()
    options.validate()
    kwargs = {**fit.window_kwargs(), **options.fit_kwargs(lt.n_exp, per_bin=True),
              **(irf_kwargs or {})}
    # MLE fallbacks are logged by the tab from meta["mle_fallback"].
    with handle.lock, warnings.catch_warnings():
        warnings.simplefilter("ignore", MLEFallbackWarning)
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


def irf_trace_kwargs(fit: FitResult | None, source: str,
                     irf: LoadedFile | None) -> tuple[dict[str, Any] | None, str | None]:
    """Per-bin IRF settings for the lifetime trace/FLIDs from the last lifetime fit.

    The per-bin fits have few photons, so they take the IRF shift, the
    background (as a rate per second and lifetime bin) and the scatter (as a
    fraction of the photons) from the reconvolution fit of the summed
    histogram. Returns ``(kwargs, None)``, or ``(None, reason)`` when there is
    no suitable lifetime fit (none, of another file, or without the IRF).
    """
    if irf is None:
        return None, "no IRF file is loaded"
    if fit is None:
        return None, "no lifetime fit has been run yet"
    if fit.source != source:
        return None, "the last lifetime fit belongs to another file"
    if not fit.fit.get("reconvolution"):
        return None, "the last lifetime fit did not use the IRF"
    if not fit.fit.get("success", False):
        return None, "the last lifetime fit with the IRF did not converge"
    duration = fit.histogram.meta.get("acquisition_duration")
    rate = (float(fit.fit["background"]) / float(duration)
            if duration else None)
    total = float(np.sum(fit.histogram.y))
    fraction = float(fit.fit.get("scatter", 0.0)) / total if total > 0 else 0.0
    return ({"irf": irf.reader, "irf_shift": float(fit.fit["shift_ns"]),
             "background_rate": rate, "scatter_fraction": fraction}, None)


def background_trace_kwargs(fit: FitResult | None,
                            source: str) -> tuple[dict[str, Any] | None, str | None]:
    """Per-bin background of the tail fits (lifetime trace/FLIDs) from the last lifetime fit.

    A time bin holds too few photons to fit its own background, so the
    per-bin tail fits take it, as a rate per second and lifetime bin, from
    the lifetime fit of the summed histogram with a fitted background.
    Returns ``(kwargs, None)``, or ``(None, reason)`` when there is no
    suitable lifetime fit (none, of another file, an IRF fit, or without a
    fitted background).
    """
    if fit is None:
        return None, "no lifetime fit has been run yet"
    if fit.source != source:
        return None, "the last lifetime fit belongs to another file"
    if fit.fit.get("reconvolution"):
        return None, "the last lifetime fit used the IRF"
    if not fit.fit.get("fit_background"):
        return None, "the last lifetime fit did not fit the background"
    if not fit.fit.get("success", False):
        return None, "the last lifetime fit did not converge"
    duration = fit.histogram.meta.get("acquisition_duration")
    if not duration:
        return None, "the fitted histogram has no acquisition duration"
    return {"background_rate": float(fit.fit["background"]) / float(duration)}, None


# ------------------------------------------------------------------ summaries
def error_key(key: str) -> str:
    """Summary key of the standard error of ``key`` (``tau1_ns`` -> ``tau1_err_ns``)."""
    return f"{key[:-3]}_err_ns" if key.endswith("_ns") else f"{key}_err"


def fit_summary(fit: dict[str, Any]) -> dict[str, Any]:
    """Scalar fit results: lifetimes (ns), amplitudes, mean lifetime, offset, quality, window.

    Per component ``i``: ``tau{i}_ns``, the amplitude fraction ``A{i}_pct``
    (``A_i / ΣA`` in %), the intensity fraction ``I{i}_pct``
    (``A_i τ_i / ΣAτ`` in %), the per-pulse amplitude ``A{i}`` in counts
    (corrected for the incomplete decay of earlier laser pulses when the fit
    did so) and, with that correction, the fitted amplitude ``A{i}_fit``.
    Standard errors are stored under :func:`error_key`. The mean lifetime is
    ``tau_int_ns`` or ``tau_amp_ns``, as chosen by the fit's ``mean_lifetime``.
    """
    quality = fit.get("quality", {})
    summary: dict[str, Any] = {}
    A, tau = np.atleast_1d(fit["A"]), np.atleast_1d(fit["tau"])
    columns = {"tau{}_ns": (tau, fit.get("tau_err")),
               "A{}_pct": (fit.get("f_amp"), fit.get("f_amp_err"), 100.0),
               "I{}_pct": (fit.get("f_int"), fit.get("f_int_err"), 100.0),
               "A{}": (A, fit.get("A_err"))}
    if fit.get("decay_correction", True) and "A_fit" in fit:
        columns["A{}_fit"] = (fit["A_fit"], fit.get("A_fit_err"))
    for i in range(A.size):
        for pattern, (values, errors, *factor) in columns.items():
            if values is None:
                continue
            scale = factor[0] if factor else 1.0
            key = pattern.format(i + 1)
            summary[key] = float(np.atleast_1d(values)[i]) * scale
            if errors is not None:
                summary[error_key(key)] = float(np.atleast_1d(errors)[i]) * scale
    kind = fit.get("mean_lifetime", "intensity")
    name = "tau_int" if kind == "intensity" else "tau_amp"
    if name in fit:
        summary[f"{name}_ns"] = float(fit[name])
        summary[f"{name}_err_ns"] = float(fit.get(f"{name}_err", np.nan))
    if fit.get("reconvolution"):
        summary["irf_shift_ns"] = float(fit["shift_ns"])
        if fit.get("fit_shift", True):
            summary["irf_shift_err_ns"] = float(fit.get("shift_err_ns", np.nan))
        if fit.get("fit_scatter"):
            summary["scatter"] = float(fit["scatter"])
            summary["scatter_err"] = float(fit.get("scatter_err", np.nan))
        summary["irf_fwhm_ns"] = float(fit.get("irf_fwhm_ns", np.nan))
    if fit.get("fit_background"):
        summary["offset"] = float(fit["background"])
        summary["offset_err"] = float(fit.get("background_err", np.nan))
    else:
        summary["offset"] = float(fit.get("offset", 0.0))
    if fit.get("prerise"):
        summary["prerise_bins"] = int(fit.get("prerise_points", 0))
    if fit.get("seam_points"):
        summary["seam_bins"] = int(fit["seam_points"])
    if "weighting" in fit:
        summary["statistic"] = statistic_text(fit)
    summary["reduced_chi_squared"] = float(quality.get("reduced_chi_squared", np.nan))
    if "bic" in quality:
        summary["bic"] = float(quality["bic"])
    summary["r_squared"] = float(quality.get("r_squared", np.nan))
    summary["success"] = bool(fit.get("success", False))
    summary["fit_start_ns"] = float(fit.get("left_bound_ns", np.nan))
    summary["fit_end_ns"] = float(np.max(fit["t_fit"])) if len(fit.get("t_fit", [])) else np.nan
    summary["fit_points"] = int(len(fit.get("t_fit", [])))
    if fit.get("lifetime_shift_ns"):
        summary["lifetime_shift_ns"] = float(fit["lifetime_shift_ns"])
    if fit.get("channels"):
        summary["channels"] = ", ".join(str(c) for c in fit["channels"])
    return summary


#: Columns of the per-component results table: (``fit_summary`` key pattern, title).
COMPONENT_COLUMNS = (("tau{}_ns", "τ (ns)"), ("A{}_pct", "A (%)"), ("I{}_pct", "I (%)"),
                     ("A{}", "A (counts)"), ("A{}_fit", "A fitted (counts)"))
#: ``fit_summary`` keys of the mean lifetimes, shown in the τ column.
_MEAN_KEYS = {"tau_int_ns": "intensity", "tau_amp_ns": "amplitude"}


@dataclass
class ComponentRow:
    """One row of the per-component table.

    ``label`` is the component number (``"1"``, ``"2"``, ...) or, for the
    mean lifetime, its symbol; ``keys`` maps a column title of
    :data:`COMPONENT_COLUMNS` to the ``fit_summary`` key shown there.
    """

    label: str
    keys: dict[str, str]
    mean: bool = False


def component_columns(summaries: Sequence[dict[str, Any]]) -> list[str]:
    """Titles of the component columns used by any of ``summaries`` (in table order)."""
    return [title for pattern, title in COMPONENT_COLUMNS
            if any(pattern.format(1) in s for s in summaries)]


def component_rows(summary: dict[str, Any]) -> list[ComponentRow]:
    """The components of one fit, then its mean lifetime (in the τ column)."""
    rows = []
    i = 1
    while f"tau{i}_ns" in summary:
        keys = {title: pattern.format(i) for pattern, title in COMPONENT_COLUMNS
                if pattern.format(i) in summary}
        rows.append(ComponentRow(str(i), keys))
        i += 1
    for key, kind in _MEAN_KEYS.items():
        if key in summary:
            rows.append(ComponentRow(f"Mean {MEAN_SYMBOLS[kind]}", {"τ (ns)": key}, mean=True))
    return rows


def quality_keys(summaries: Sequence[dict[str, Any]]) -> list[str]:
    """``fit_summary`` keys of the second table: everything not per component.

    Standard errors are left out; they are shown with their value.
    """
    used = {key for s in summaries for row in component_rows(s) for key in row.keys.values()}
    keys = list(dict.fromkeys(k for s in summaries for k in s))
    errors = {error_key(k) for k in keys}
    return [k for k in keys if k not in used and k not in errors]


def summary_text(summary: dict[str, Any], key: str) -> str:
    """A summary value as shown in the tables: ``value ± error`` where there is one."""
    value = summary.get(key)
    if value is None:
        return ""
    if error_key(key) in summary:
        return format_uncertainty(value, summary[error_key(key)])
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def mle_fallback_text(fit: dict[str, Any]) -> str | None:
    """Log line when a lifetime fit's Poisson MLE fell back to least squares."""
    if fit.get("mle_converged") is not False:
        return None
    return ("the Poisson MLE did not converge, so the Poisson least-squares fit is shown "
            "(its BIC is not comparable with MLE fits). Check the fit window and the "
            "background.")


def bin_fallback_text(signal: Signal) -> str | None:
    """Log line when per-bin Poisson MLE fits fell back to least squares (``None`` if none)."""
    fallback = np.asarray(signal.meta.get("mle_fallback", []), dtype=bool)
    if not fallback.any():
        return None
    fitted = int(np.count_nonzero(np.isfinite(np.asarray(signal.y, dtype=float))))
    return (f"the Poisson MLE did not converge in {int(fallback.sum())} of {fitted} fitted "
            "time bins; the Poisson least-squares result is used there.")


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


def mean_text(fit: dict[str, Any]) -> str | None:
    """``"τ_int = 3.42 ± 0.05 ns"`` or ``"τ_amp = …"``: the fit's chosen mean lifetime."""
    kind = fit.get("mean_lifetime", "intensity")
    name = "tau_int" if kind == "intensity" else "tau_amp"
    if name not in fit:
        return None
    value = format_uncertainty(fit[name], fit.get(f"{name}_err", np.nan))
    return f"{MEAN_SYMBOLS[kind]} = {value} ns"


def quality_text(fit: dict[str, Any]) -> str:
    """Short goodness-of-fit string for plot titles, e.g. ``"χ²ᵣ = 1.08, R² = 0.998"``."""
    quality = fit.get("quality", {})
    chi = quality.get("reduced_chi_squared", np.nan)
    r2 = quality.get("r_squared", np.nan)
    text = f"χ²ᵣ = {chi:.3g}, R² = {r2:.4f}"
    if fit.get("mle_converged") is False:
        text += " | MLE did not converge: Poisson least squares"
    return text


#: Names of the fit statistics in the results table.
STATISTIC_NAMES = {"mle": "Poisson MLE", "poisson": "Poisson least squares",
                   "none": "Least squares"}


def statistic_text(fit: dict[str, Any]) -> str:
    """The fit statistic actually used, e.g. ``"Poisson least squares (MLE did not converge)"``."""
    name = STATISTIC_NAMES.get(fit.get("weighting", ""), str(fit.get("weighting", "")))
    if fit.get("mle_converged") is False:
        name += " (MLE did not converge)"
    return name


# ------------------------------------------------------------ BIC comparison
@dataclass
class BicComparison:
    """BIC of a lifetime fit relative to the previous fit (see :func:`compare_bic`).

    ``verdict`` is ``"better"`` (ΔBIC < −threshold), ``"worse"`` (ΔBIC >
    threshold, e.g. an added component that only fits noise), ``"same"``,
    ``"not comparable"`` (the previous fit used other data, window or
    statistic) or ``"first"`` (no previous fit). ``delta`` is
    ``BIC(new) − BIC(previous)``; ``previous_n_exp`` the components of the
    previous fit.
    """

    bic: float
    verdict: str
    delta: float = float("nan")
    previous_n_exp: int | None = None
    threshold: float = BIC_THRESHOLD

    @property
    def colour(self) -> str:
        """Verdict colour key: ``"good"`` (green), ``"bad"`` (red) or ``"neutral"`` (gray)."""
        return {"better": "good", "worse": "bad"}.get(self.verdict, "neutral")

    def text(self) -> str:
        """E.g. ``"BIC = 2131.6, ΔBIC = −45.2 vs. 1 exp: better"``."""
        base = f"BIC = {self.bic:.1f}"
        if self.verdict == "first":
            return base
        if self.verdict == "not comparable":
            return f"{base} (previous fit not comparable: other data, window or weighting)"
        verdicts = {"better": "better fit", "worse": "worse fit (overfitting?)",
                    "same": "about the same"}
        return (f"{base}, ΔBIC = {self.delta:+.1f} vs. {self.previous_n_exp} exp: "
                f"{verdicts[self.verdict]} (threshold {self.threshold:g})")

    def short_text(self) -> str:
        """Compact form for the fit table, e.g. ``"2131.6 (Δ −45.2)"``."""
        if self.verdict in ("first", "not comparable"):
            return f"{self.bic:.1f}"
        return f"{self.bic:.1f} (Δ {self.delta:+.1f})"


def same_fit_data(a: FitResult, b: FitResult) -> bool:
    """Whether two fits used the same counts, fit window and statistic (BIC comparable)."""
    qa, qb = a.fit.get("quality", {}), b.fit.get("quality", {})
    return (a.source == b.source
            and a.fit.get("weighting") == b.fit.get("weighting")
            and qa.get("bic_statistic") == qb.get("bic_statistic")
            and np.array_equal(np.asarray(a.histogram.y), np.asarray(b.histogram.y))
            and np.array_equal(np.asarray(a.fit.get("t_fit", [])),
                               np.asarray(b.fit.get("t_fit", []))))


def compare_bic(result: FitResult, previous: Sequence[FitResult],
                threshold: float = BIC_THRESHOLD) -> BicComparison | None:
    """Compare the BIC of ``result`` with the matching fit among ``previous``.

    The match is the previous fit of the same ROI (or the first one) if it was
    made on identical data (:func:`same_fit_data`); only then is ΔBIC
    meaningful. ``None`` if ``result`` has no BIC (e.g. a failed fit).
    """
    bic = result.fit.get("quality", {}).get("bic")
    if bic is None or not np.isfinite(bic):
        return None
    if not previous:
        return BicComparison(float(bic), "first", threshold=threshold)
    same_roi = [p for p in previous if p.roi_index == result.roi_index]
    candidate = (same_roi or list(previous))[0]
    old = candidate.fit.get("quality", {}).get("bic")
    if old is None or not np.isfinite(old) or not same_fit_data(result, candidate):
        return BicComparison(float(bic), "not comparable", threshold=threshold)
    delta = float(bic) - float(old)
    verdict = "better" if delta < -threshold else "worse" if delta > threshold else "same"
    return BicComparison(float(bic), verdict, delta, int(candidate.params.n_exp), threshold)


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
                 "detector_channels": ("all" if result.channels is None
                                       else list(result.channels)),
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
                   units={"x": "ns", "y": "counts"})
    residuals = fit.get("quality", {}).get("weighted_residuals")
    signals = [data, curve]
    if fit.get("reconvolution") and fit.get("irf_shifted") is not None:
        signals.append(Signal(fit["irf_t_ns"], np.asarray(fit["irf_shifted"], dtype=float),
                              label=f"{prefix} IRF (shifted, unit area)".strip(),
                              units={"x": "ns", "y": "fraction"}))
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
    if result.extra.get("irf"):
        params["irf"] = result.extra["irf"]
    if result.extra.get("fit_background"):
        params["fit_background"] = True
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


def bin_runs(times: Sequence[float], step: float) -> list[tuple[float, float]]:
    """``(first, last)`` time of every run of neighbouring bins in ``times``.

    ``times`` are sorted bin positions and ``step`` the bin width; bins less
    than 1.5 widths apart belong to one run (used to mark the seam bins of a
    fit as a few spans instead of one line per bin).
    """
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(times) > 1.5 * abs(step)) + 1
    return [(float(run[0]), float(run[-1])) for run in np.split(times, breaks)]


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


def _result_summary(result: FitResult) -> dict[str, Any]:
    """:func:`fit_summary` plus the warnings and the BIC comparison of the GUI."""
    summary = {**fit_summary(result.fit), "histogram": describe_meta(result.histogram.meta)}
    if result.fit.get("reconvolution"):
        summary["irf_background_subtracted"] = float(result.fit.get("irf_background", 0.0))
    if result.fit.get("warnings"):
        summary["warnings"] = list(result.fit["warnings"])
    comparison = result.extra.get("bic_comparison")
    if comparison is not None:
        summary["bic_comparison"] = asdict(comparison)
    return summary


def fit_payload(result: FitResult) -> ExportPayload:
    hist = result.histogram
    meta = build_metadata(
        "PTU lifetime fit", sources=[result.source],
        parameters=_fit_parameters(result),
        results=_result_summary(result),
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
        summaries[name] = _result_summary(result)
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
    amps_fit = signal.meta.get("amplitudes_fit")
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
        if amps_fit is not None and np.shape(amps_fit) == amps.shape:
            columns.append(Signal(signal.x, np.asarray(amps_fit)[:, i],
                                  label=f"A{i + 1}_fit", units={"x": "s", "y": ""}))
    parameters = _params(("time_trace", params), ("fit_window", fit), ("lifetime_trace", lt),
                         ("fit_options", options or FitOptions()))
    if signal.meta.get("reconvolution"):
        parameters["irf"] = {"shift_ns": signal.meta.get("irf_shift_ns")}
    elif signal.meta.get("background_rate") is not None:
        # Background of the per-bin tail fits, from the last lifetime fit.
        parameters["background_rate_per_s_and_bin"] = float(signal.meta["background_rate"])
    if lt.roi is not None and 0 <= lt.roi < len(rois):
        parameters["roi"] = dict(rois[lt.roi])
    meta = build_metadata(
        "PTU lifetime trace", sources=[source], parameters=parameters,
        note="Empty lifetime cells are time bins without photons, outside the ROI or "
             "with a failed fit.",
    )
    return ExportPayload(meta, columns, default_name=signal.label.replace(" ", "_"))
