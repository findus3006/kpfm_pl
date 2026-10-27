"""Correlate workflow: align a KPFM signal with a PL time trace (no Qt).

Two steps, mirroring ``examples/example_correlate.py``:

1. :func:`compute_triggers` builds the PL time trace (``PTU``) and the
   bin-averaged KPFM signal (``Zurich``) and locates the laser triggers in
   both. :func:`trigger_problems` lists anything that would make the
   alignment fail or be questionable.
2. :func:`compute_match` aligns both signals with ``Match`` and cuts out the
   window chosen by :class:`WindowParams`; :func:`extract_window` re-cuts an
   existing alignment cheaply (e.g. after dragging a new window).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ...core import Signal
from ...match import Match
from ...utils import TriggerPoint
from ..export import ExportPayload, build_metadata
from . import ptu as ptu_model
from .files import LoadedFile

__all__ = [
    "ZurichParams",
    "MatchParams",
    "WindowParams",
    "TriggerResult",
    "MatchResult",
    "compute_triggers",
    "trigger_problems",
    "compute_match",
    "extract_window",
    "match_summary",
    "trigger_payload",
    "aligned_payload",
]

TRIGGER_METHODS = (("slope", "Slope (steepest step)"), ("threshold", "Threshold (10 % level)"))
FLANKS = (("rising", "Rising"), ("falling", "Falling"), ("both", "Both"))
#: Kinds of window bounds; "trigger" uses a 1-based trigger number, "time" seconds.
BOUND_KINDS = (("first", "First trigger"), ("last", "Last trigger"),
               ("trigger", "Trigger number"), ("time", "Time"),
               ("overlap", "Edge of overlap"))


@dataclass
class ZurichParams:
    """Arguments of ``Zurich.average`` and ``Zurich.laser_triggers``.

    ``None`` stream names select the first (KPFM) and second (trigger) stream
    of the file, as in the examples.
    """

    kpfm_stream: str | None = None
    trigger_stream: str | None = None
    bintime: float = 0.05
    method: str = "slope"
    flank: str = "rising"
    slope_threshold: float = 0.1
    units: str = "V"


@dataclass
class MatchParams:
    """Arguments of ``Match``; ``align`` is the 0-based trigger position."""

    align: int = 0
    dt: float | None = None
    kpfm_smoothing: int = 20
    timetrace_smoothing: int = 1


@dataclass
class WindowParams:
    """Bounds passed to ``Match.extract``.

    ``start``/``stop`` are one of the keys of :data:`BOUND_KINDS`. For
    ``"trigger"`` the 1-based ``*_trigger`` number is used, for ``"time"``
    the ``*_time`` in seconds on the aligned (KPFM) time axis.
    """

    start: str = "first"
    start_trigger: int = 1
    start_time: float = 0.0
    stop: str = "last"
    stop_trigger: int = 1
    stop_time: float = 0.0


@dataclass
class TriggerResult:
    """Inputs of the alignment: both signals and their triggers.

    ``trigger_signal`` is a min/max envelope of the raw trigger stream, for
    the preview plot only.
    """

    kpfm: Signal
    trigger_signal: Signal
    timetrace: Signal
    kpfm_triggers: list[TriggerPoint] | None
    timetrace_triggers: list[TriggerPoint] | None
    trace_params: ptu_model.TraceParams
    zurich_params: ZurichParams
    ptu_source: str
    zurich_source: str


@dataclass
class MatchResult:
    """An alignment plus the currently extracted window."""

    triggers: TriggerResult
    match: Match
    window: dict[str, Signal]
    start: float
    stop: float
    match_params: MatchParams
    window_params: WindowParams


def envelope(signal: Signal, max_points: int = 20_000) -> Signal:
    """Min/max envelope of ``signal`` for plotting (display only).

    Averaging would flatten the short trigger pulses; the envelope keeps every
    pulse visible while reducing ~10^6 samples to ``2 * max_points``.
    """
    n = len(signal)
    block = max(1, n // max_points)
    m = (n // block) * block
    x = signal.x[:m].reshape(-1, block)
    y = signal.y[:m].reshape(-1, block)
    return Signal(np.repeat(x[:, 0], 2), np.column_stack([y.min(1), y.max(1)]).ravel(),
                  label=f"{signal.label} (envelope)", units=dict(signal.units),
                  meta={"envelope_block": block})


def _resolve_streams(names: list[str], params: ZurichParams) -> tuple[str, str]:
    if not names:
        raise ValueError("The Zurich file contains no data streams.")
    kpfm = params.kpfm_stream or names[0]
    trigger = params.trigger_stream or (names[1] if len(names) > 1 else None)
    if trigger is None:
        raise ValueError("The Zurich file has only one stream; a separate laser "
                         "trigger stream is needed for the alignment.")
    for name in (kpfm, trigger):
        if name not in names:
            raise ValueError(f"Stream {name!r} is not in the Zurich file "
                             f"(available: {', '.join(names)}).")
    return kpfm, trigger


def compute_triggers(ptu: LoadedFile, zurich: LoadedFile,
                     trace_params: ptu_model.TraceParams,
                     zurich_params: ZurichParams) -> TriggerResult:
    """Build both signals and locate the laser triggers in each.

    The Zurich triggers are mapped onto the averaged KPFM axis
    (``reference=``), so their indices line up with the signal ``Match`` uses.
    """
    trace = ptu_model.compute_trace(ptu, trace_params)
    with zurich.lock:
        z = zurich.reader
        kpfm_stream, trigger_stream = _resolve_streams(z.stream_names(), zurich_params)
        kpfm = z.average(kpfm_stream, bintime=zurich_params.bintime, units=zurich_params.units)
        trigger_signal = envelope(z.signal(trigger_stream, units=zurich_params.units))
        kpfm_triggers = z.laser_triggers(trigger_stream, method=zurich_params.method,
                                         flank=zurich_params.flank, reference=kpfm,
                                         slope_threshold=zurich_params.slope_threshold)
    resolved = ZurichParams(**{**asdict(zurich_params), "kpfm_stream": kpfm_stream,
                               "trigger_stream": trigger_stream})
    return TriggerResult(kpfm, trigger_signal, trace.signal, kpfm_triggers, trace.triggers,
                         trace_params, resolved, str(ptu.path), str(zurich.path))


def trigger_problems(result: TriggerResult) -> list[str]:
    """Human-readable problems with the detected triggers (empty if fine)."""
    problems = []
    n_k = len(result.kpfm_triggers or [])
    n_t = len(result.timetrace_triggers or [])
    if n_k == 0:
        problems.append("No laser trigger found in the Zurich trigger stream. Check the "
                        "trigger stream, method and flank (legacy files need 'Both').")
    if n_t == 0:
        problems.append("No laser trigger found in the PTU file. Legacy files without "
                        "markers need 'Laser markers' unchecked.")
    if n_k and n_t and n_k != n_t:
        problems.append(f"The Zurich file has {n_k} triggers but the PTU file has {n_t}. "
                        "Alignment uses the same trigger number in both lists; check that "
                        "the chosen trigger corresponds (or change the flank).")
    return problems


def _bound(match: Match, kind: str, trigger: int, time: float, name: str):
    if kind == "overlap":
        return None
    if kind == "time":
        return float(time)
    n = len(match.triggers)
    if n == 0:
        raise ValueError(f"The window {name} refers to a trigger, but no triggers lie on "
                         "the aligned time axis.")
    if kind == "first":
        return match.triggers[0]
    if kind == "last":
        return match.triggers[-1]
    if kind == "trigger":
        if not 1 <= trigger <= n:
            raise ValueError(f"Window {name}: trigger number {trigger} does not exist "
                             f"(there are {n} triggers).")
        return match.triggers[trigger - 1]
    raise ValueError(f"Unknown window bound {kind!r}.")


def extract_window(match: Match, window: WindowParams) -> tuple[dict[str, Signal], float, float]:
    """Cut ``match`` to ``window``; return the signals and the bounds in seconds."""
    start = _bound(match, window.start, window.start_trigger, window.start_time, "start")
    stop = _bound(match, window.stop, window.stop_trigger, window.stop_time, "end")
    extracted = match.extract(start, stop)
    x = extracted["kpfm"].x
    return extracted, float(x[0]), float(x[-1])


def compute_match(triggers: TriggerResult, params: MatchParams,
                  window: WindowParams) -> MatchResult:
    """Align both signals on their triggers and extract the window."""
    n_k = len(triggers.kpfm_triggers or [])
    n_t = len(triggers.timetrace_triggers or [])
    if not n_k or not n_t:
        raise ValueError("Both files need at least one laser trigger to be aligned. "
                         + " ".join(trigger_problems(triggers)))
    n_common = min(n_k, n_t)
    if not 0 <= params.align < n_common:
        raise ValueError(f"Cannot align on trigger number {params.align + 1}: the Zurich "
                         f"file has {n_k} and the PTU file {n_t} triggers.")
    try:
        match = Match(triggers.kpfm, triggers.timetrace,
                      kpfm_triggers=triggers.kpfm_triggers,
                      timetrace_triggers=triggers.timetrace_triggers,
                      align=params.align, dt=params.dt,
                      kpfm_smoothing=params.kpfm_smoothing,
                      timetrace_smoothing=params.timetrace_smoothing)
    except ValueError as exc:
        if "overlapping" in str(exc):
            raise ValueError("After aligning on the chosen trigger the KPFM signal and the "
                             "time trace do not overlap in time. Check that the triggers "
                             "belong to the same laser event.") from exc
        raise
    extracted, start, stop = extract_window(match, window)
    return MatchResult(triggers, match, extracted, start, stop, params, window)


# ------------------------------------------------------------------ summaries
def match_summary(result: MatchResult) -> dict[str, Any]:
    """Scalar alignment results for the results table and the export header."""
    m = result.match
    in_window = result.window["kpfm"].meta.get("triggers", [])
    return {
        "time_shift_s": float(m.offset),
        "grid_step_s": float(m.dt),
        "kpfm_triggers": len(result.triggers.kpfm_triggers or []),
        "timetrace_triggers": len(result.triggers.timetrace_triggers or []),
        "aligned_on_trigger": result.match_params.align + 1,
        "window_start_s": result.start,
        "window_end_s": result.stop,
        "window_duration_s": result.stop - result.start,
        "triggers_in_window": len(in_window),
    }


def _parameters(triggers: TriggerResult, match: MatchParams | None = None,
                window: WindowParams | None = None) -> dict[str, Any]:
    params = {"time_trace": asdict(triggers.trace_params),
              "zurich": asdict(triggers.zurich_params)}
    params["time_trace"].pop("dark_counts", None)
    params["time_trace"].pop("tstart", None)
    params["time_trace"].pop("tend", None)
    if match is not None:
        params["match"] = asdict(match)
        params["match"]["align_trigger_number"] = params["match"].pop("align") + 1
    if window is not None:
        params["window"] = asdict(window)
    return params


def _times(points) -> list[float]:
    return [float(p["time"]) for p in points or []]


def trigger_payload(result: TriggerResult) -> ExportPayload:
    """The two input signals (own time axes) with their trigger times."""
    meta = build_metadata(
        "KPFM/PL trigger detection", sources=[result.zurich_source, result.ptu_source],
        parameters=_parameters(result),
        results={"kpfm_trigger_times_s": _times(result.kpfm_triggers),
                 "timetrace_trigger_times_s": _times(result.timetrace_triggers)},
        note="Signals on their own, unaligned time axes.",
    )
    return ExportPayload(meta, [result.kpfm, result.timetrace], default_name="triggers")


def aligned_payload(result: MatchResult) -> ExportPayload:
    """The extracted window: KPFM and time trace on one shared time axis."""
    kpfm, trace = result.window["kpfm"], result.window["timetrace"]
    meta = build_metadata(
        "KPFM/PL correlation", sources=[result.triggers.zurich_source,
                                        result.triggers.ptu_source],
        parameters=_parameters(result.triggers, result.match_params, result.window_params),
        results={**match_summary(result),
                 "trigger_times_in_window_s": _times(kpfm.meta.get("triggers"))},
        note="Time axis of the KPFM signal; the time trace was shifted by time_shift_s. "
             "Empty cells lie outside a signal's own time span. Values are not scaled.",
    )
    stem = kpfm.label.split(" ")[0] if kpfm.label else "correlation"
    return ExportPayload(meta, [kpfm, trace], default_name=f"{stem}_correlation")


def scaled(y: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Min-max scale ``y`` with the finite range of ``reference`` (display only)."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.asarray(y, dtype=float)
    lo, span = float(ref.min()), float(ref.max() - ref.min())
    return (np.asarray(y, dtype=float) - lo) / span if span else np.zeros_like(y, dtype=float)
