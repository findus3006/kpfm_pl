"""Antibunching workflow: detector trace, region, second-order correlation (no Qt).

The correlation ``g2(τ)`` of two detectors (``PTU.g2``) needs a T2 file:
its photons carry time tags at the full timing resolution. The regions are
chosen on the trace of the two detectors (``PTU.detector_trace``), which
neither uses nor changes the detector selection and the cached time trace of
the *PL lifetime* tab, so both tabs can work on the same file.

A region is the whole time window, the on state (bins with at least a given
percentage of the maximum count of the window) or an ROI drawn on the trace.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ...core import Signal
from ..export import ExportPayload, build_metadata
from .files import LoadedFile
from .ptu import format_uncertainty

__all__ = [
    "G2TraceParams",
    "G2Params",
    "G2Result",
    "unavailable_reason",
    "detector_options",
    "describe_g2_file",
    "compute_detector_trace",
    "region_options",
    "selection",
    "compute_g2",
    "result_rows",
    "g2_title",
    "trace_payload",
    "g2_payload",
]

#: Regions besides the ROIs (``"roi:<n>"``, 1-based).
REGIONS = (("all", "Whole time window"), ("on", "On state"))


@dataclass
class G2TraceParams:
    """The trace the region is chosen on: bin time and the two detectors."""

    bintime: float = 0.01
    channel_a: int = 1
    channel_b: int = 2

    def channels(self) -> tuple[int, int]:
        return (int(self.channel_a), int(self.channel_b))


@dataclass
class G2Params:
    """Region and correlation settings of ``PTU.g2`` (times in the GUI units).

    ``region`` is ``"all"``, ``"on"`` or ``"roi"`` (then ``roi``, 0-based).
    ``tstart``/``tend`` (s, ``tend=None``: end of file) limit the time window
    of ``"all"`` and ``"on"``; ``on_percent`` is the on-state threshold in %
    of the largest count in that window. The lag settings are in ns; ``None``
    means automatic (see ``PTU.g2``).
    """

    region: str = "all"
    roi: int = 0
    on_percent: float = 60.0
    tstart: float = 0.0
    tend: float | None = None
    max_lag_ns: float | None = None
    lag_bin_ns: float | None = None
    period_ns: float | None = None
    delay_ns: float | None = None
    window_ns: float | None = None

    def validate(self) -> None:
        if self.region not in ("all", "on", "roi"):
            raise ValueError(f"Unknown region {self.region!r}.")
        if not 0 < self.on_percent <= 100:
            raise ValueError("The on-state threshold must lie between 0 and 100 %.")
        if self.tend is not None and self.tend <= self.tstart:
            raise ValueError("The time window must end after it starts.")
        for name in ("max_lag_ns", "lag_bin_ns", "period_ns", "window_ns"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name.replace('_ns', '').replace('_', ' ')} must be "
                                 "positive.")

    def g2_kwargs(self) -> dict[str, Any]:
        """The lag settings as ``PTU.g2`` arguments (seconds)."""
        names = {"max_lag_ns": "max_lag", "lag_bin_ns": "lag_bin", "period_ns": "period",
                 "delay_ns": "delay", "window_ns": "window"}
        return {arg: getattr(self, key) * 1e-9 for key, arg in names.items()
                if getattr(self, key) is not None}


@dataclass
class G2Result:
    """One computed correlation with everything needed to plot and export it."""

    signal: Signal
    trace_params: G2TraceParams
    params: G2Params
    source: str
    region: dict[str, float | None]
    region_name: str
    rois: list[dict[str, float]] = field(default_factory=list)


def unavailable_reason(handle: LoadedFile | None) -> str | None:
    """Why ``g2`` cannot be computed for this file (``None`` if it can)."""
    if handle is None:
        return "Open a .ptu file to start."
    ptu = handle.reader
    if getattr(ptu, "is_cw", False):
        return "This continuous-wave file holds no photon records."
    if ptu.measurement_mode != "T2":
        return (f"This is a {ptu.measurement_mode} file. The correlation needs a T2 file "
                "(time tags of every photon at full resolution).")
    if len(ptu.channels) < 2:
        return (f"This T2 file holds photons of one detector only (channel "
                f"{', '.join(map(str, ptu.channels)) or '–'}); the correlation needs two.")
    return None


def detector_options(handle: LoadedFile | None) -> list[tuple[int, str]]:
    if handle is None or getattr(handle.reader, "is_cw", False):
        return [(1, "Detector 1"), (2, "Detector 2")]
    return [(int(c), f"Detector {c}") for c in handle.reader.channels]


def describe_g2_file(handle: LoadedFile) -> str:
    """One line on the file: device, detectors and the laser period, if known."""
    ptu = handle.reader
    text = (f"{ptu.device}, {ptu.measurement_mode} mode, detectors "
            f"{', '.join(map(str, ptu.channels)) or '–'}")
    if ptu.measurement_mode == "T2":
        if ptu.has_lifetime and np.isfinite(ptu.rep_time):
            text += f"; laser period {ptu.rep_time * 1e9:.6g} ns from the sync"
        else:
            text += ("; no laser sync: enter the laser period for the pulsed g2(0), "
                     "otherwise g2(0) of continuous excitation is computed")
    return text + "."


def compute_detector_trace(handle: LoadedFile, params: G2TraceParams) -> Signal:
    """Photons per time bin of the two detectors (``PTU.detector_trace``)."""
    with handle.lock:
        return handle.reader.detector_trace(params.bintime, params.channels())


def region_options(rois: Sequence[dict[str, float]]) -> list[tuple[str, str]]:
    return [*REGIONS, *((f"roi:{i + 1}", f"ROI {i + 1}") for i in range(len(rois)))]


def selection(params: G2Params, trace: Signal,
              rois: Sequence[dict[str, float]]) -> tuple[dict[str, float | None], str]:
    """``({x_min, x_max, y_min, y_max}, name)`` of the region for ``PTU.g2``."""
    if params.region == "roi":
        if not 0 <= params.roi < len(rois):
            raise ValueError(f"ROI {params.roi + 1} does not exist ({len(rois)} ROI(s) drawn).")
        roi = rois[params.roi]
        return ({key: float(roi[key]) for key in ("x_min", "x_max", "y_min", "y_max")},
                f"ROI {params.roi + 1}")
    bounds: dict[str, float | None] = {"x_min": float(params.tstart),
                                       "x_max": None if params.tend is None
                                       else float(params.tend),
                                       "y_min": None, "y_max": None}
    if params.region == "all":
        return bounds, "whole time window"
    x = np.asarray(trace.x, dtype=float)
    inside = (x + float(trace.meta["bintime"]) > params.tstart) & (
        x < (np.inf if params.tend is None else params.tend))
    counts = np.asarray(trace.y, dtype=float)[inside]
    if not counts.size or not np.max(counts) > 0:
        raise ValueError("The time window holds no photons.")
    bounds["y_min"] = params.on_percent / 100.0 * float(np.max(counts))
    return bounds, f"on state (≥ {params.on_percent:g} % of the maximum)"


def compute_g2(handle: LoadedFile, trace_params: G2TraceParams, params: G2Params,
               rois: Sequence[dict[str, float]] = (),
               progress: Callable[[int, int], Any] | None = None) -> G2Result:
    """Correlate the two detectors in the chosen region (``PTU.g2``)."""
    params.validate()
    if trace_params.channel_a == trace_params.channel_b:
        raise ValueError("Choose two different detectors.")
    with handle.lock:
        ptu = handle.reader
        trace = ptu.detector_trace(trace_params.bintime, trace_params.channels())
        bounds, name = selection(params, trace, rois)
        signal = ptu.g2(bintime=trace_params.bintime, channels=trace_params.channels(),
                        progress=progress, **bounds, **params.g2_kwargs())
    return G2Result(signal, trace_params, params, str(handle.path), bounds, name,
                    [dict(r) for r in rois])


def _ns(value: float | None) -> str:
    return "–" if value is None else f"{value * 1e9:.4g} ns"


def result_rows(result: G2Result) -> list[tuple[str, str, str]]:
    """``(label, value, tooltip)`` rows of the results table."""
    meta = result.signal.meta
    a, b = meta["channels"]
    pulsed = meta["method"] == "pulsed"
    rows = [
        ("g2(0)", format_uncertainty(meta["g2_0"], meta["g2_0_err"]),
         "Pulsed: centre-peak area / mean side-peak area. Continuous: normalised g2 at the "
         "delay. 0: single-photon emitter, 1: uncorrelated light, 1 − 1/N for N equal "
         "emitters; < 0.5 proves a single emitter. ± Poisson error of the areas."),
        ("Method", "pulsed (peak areas)" if pulsed else "continuous (normalised g2)",
         "Pulsed when a laser period is known (from the sync or entered), otherwise "
         "continuous excitation."),
        ("Region", result.region_name, "Photons used: time bins of the detector trace."),
        ("Duration", f"{meta['duration']:.4g} s ({meta['selected_bins']} bins)",
         "Total length of the selected time bins."),
        (f"Photons {a} / {b}", f"{meta['photons'][a]} / {meta['photons'][b]}",
         "Photons of each detector in the region."),
        (f"Rate {a} / {b}", f"{meta['rates'][a]:.4g} / {meta['rates'][b]:.4g} /s",
         "Count rate of each detector in the region."),
    ]
    if pulsed:
        rows += [
            ("Centre peak", f"{meta['centre_area']:.0f}",
             "Coincidences within the window around the delay (same laser pulse)."),
            ("Side peaks", f"{meta['mean_side_area']:.1f} (mean of {meta['n_side_peaks']})",
             "Mean coincidences per side peak (photons of different pulses)."),
            ("Laser period", _ns(meta["period"]), "Spacing of the peaks."),
        ]
    rows += [
        ("Delay", _ns(meta["delay"]), "Lag of the centre peak (detector and cable delays); "
                                       "automatic: found from the peak comb."),
        ("Window", f"± {_ns(meta['window'])}", "Half-width of the area around each peak."),
        ("Lag bin", _ns(meta["lag_bin"]), "Width of one lag bin."),
        ("Max lag", _ns(meta["max_lag"]), "Largest lag of the correlogram."),
    ]
    return rows


def g2_title(result: G2Result) -> str:
    meta = result.signal.meta
    return (f"{result.signal.label}, {result.region_name}\n"
            f"g2(0) = {format_uncertainty(meta['g2_0'], meta['g2_0_err'])} ({meta['method']})")


def trace_payload(handle: LoadedFile, trace: Signal, params: G2TraceParams) -> ExportPayload:
    metadata = build_metadata("PTU detector trace (antibunching tab)", sources=[handle.path],
                              parameters=asdict(params))
    return ExportPayload(metadata, signals=[trace],
                         default_name=f"{handle.stem}_detectors_trace")


def g2_payload(result: G2Result) -> ExportPayload:
    """Coincidences and normalised g2 against the lag in ns."""
    s = result.signal
    lag_ns = np.asarray(s.x, dtype=float) * 1e9
    coincidences = Signal(lag_ns, s.y, label="coincidences",
                          units={"x": "ns", "y": "counts"})
    normalised = Signal(lag_ns, s.meta["g2"], label="g2 normalised",
                        units={"x": "ns", "y": ""})
    meta = s.meta
    results = {key: meta[key] for key in ("g2_0", "g2_0_err", "method", "period", "delay",
                                          "window", "lag_bin", "max_lag", "photons",
                                          "rates", "duration", "selected_bins",
                                          "expected_per_bin")}
    for key in ("centre_area", "side_areas", "mean_side_area", "n_side_peaks"):
        if key in meta:
            results[key] = meta[key]
    metadata = build_metadata(
        "PTU second-order correlation g2 (antibunching)", sources=[result.source],
        parameters={"trace": asdict(result.trace_params), "g2": asdict(result.params),
                    "region": result.region, "region_name": result.region_name,
                    "rois": result.rois},
        results=results)
    a, b = meta["channels"]
    stem = s.label.split(" g2")[0]
    return ExportPayload(metadata, signals=[coincidences, normalised],
                         default_name=f"{stem}_g2_{a}x{b}")
