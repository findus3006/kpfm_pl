"""Align a PL time trace with a KPFM signal for joint analysis.

The :class:`Match` class takes two :class:`~kpfm_pl.core.Signal` traces plus the
laser triggers of each, then:

1. shifts the time axes so the *left* triggers coincide,
2. amplitude-scales the two signals with a selectable method (identity by
   default, ``"feature"`` min-max to ``[0, 1]``, ``"zscore"`` standardization,
   or the
   quantitative two-point ``"trigger"`` calibration onto the KPFM scale),
3. resamples both onto a shared, uniform time grid.

Everything is returned as dicts/:class:`Signal` objects; there is no plotting.
Statistical helpers (Pearson correlation, cross-correlation/lag) operate on the
trigger-bounded overlap region.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import stats

from .core import Signal
from .utils import (
    TriggerPair,
    derivative,
    feature_scale_params,
    resample,
    smooth,
    zscore_params,
)

__all__ = ["Match"]


_TriggerSpec = TriggerPair | float


def _trigger_times(triggers: _TriggerSpec) -> tuple[float, float]:
    """Left/right trigger times.

    Accepts either the full dict (``{"left": {"index", "time"}, "right": ...}``)
    or a single ``float`` time, in which case the right trigger equals the left.
    """
    if isinstance(triggers, dict):
        return triggers["left"]["time"], triggers["right"]["time"]
    t = float(triggers)
    return t, t


def _validate_signal(signal: Signal, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return numeric arrays after validating an input signal for alignment."""
    if not isinstance(signal, Signal):
        raise TypeError(f"{name} must be a Signal, got {type(signal).__name__}.")
    x = np.asarray(signal["x"], dtype=float)
    y = np.asarray(signal["y"], dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError(
            f"{name} must contain equal-length one-dimensional x/y data."
        )
    if x.size < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} time axis must contain only finite values.")
    if not np.all(np.diff(x) > 0):
        raise ValueError(f"{name} time axis must be strictly increasing.")
    return x, y


def _validated_trigger_times(triggers: _TriggerSpec, name: str) -> tuple[float, float]:
    """Validate a single trigger time or a shared left/right trigger pair."""
    try:
        left, right = _trigger_times(triggers)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite time or a left/right trigger dictionary."
        ) from exc
    left, right = float(left), float(right)
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError(f"{name} trigger times must be finite.")
    if right < left:
        raise ValueError(f"{name} right trigger must not precede its left trigger.")
    return left, right


def _robust_trigger_value(y: np.ndarray, index: int, window: int) -> float:
    """Trigger-point amplitude, optionally median-filtered over a window.

    ``window <= 1`` returns the single sample at ``index``; larger windows take
    the median over ``index +/- window`` samples to suppress noise on the anchor.
    """
    if window <= 1:
        return float(y[index])
    lo = max(index - window, 0)
    hi = min(index + window + 1, len(y))
    return float(np.median(y[lo:hi]))


_SCALING_INFO: dict[str, dict[str, str]] = {
    "feature": {
        "k_label": "KPFM (aligned, normalized)",
        "t_label": "time trace (aligned, normalized)",
        "k_unit": "1",
        "t_unit": "1",
    },
    "zscore": {
        "k_label": "KPFM (aligned, z-score)",
        "t_label": "time trace (aligned, z-score)",
        "k_unit": "1",
        "t_unit": "1",
    },
    "trigger": {
        "k_label": "KPFM (aligned)",
        "t_label": "time trace (aligned, scaled)",
    },
}


class Match:
    """Align and co-scale a KPFM signal and a PL time trace.

    Parameters
    ----------
    kpfm, timetrace :
        The two :class:`Signal` traces to align.
    kpfm_triggers, timetrace_triggers :
        Either a trigger dict (``{"left": {"index", "time"}, "right": ...}``) as
        returned by :meth:`~kpfm_pl.zurich.Zurich.laser_triggers` and
        :meth:`~kpfm_pl.ptu.PTU.laser_triggers`, or a single ``float`` giving the
        time of one trigger. With a single ``float`` the right trigger equals the
        left (so only the time-shift/resampling is well defined) and
        ``scaling="trigger"`` is not available.
    n_points :
        Number of samples in the shared resampled grid. ``None`` uses the finer
        of the two native resolutions over the overlap.
    kpfm_smoothing, timetrace_smoothing :
        Optional moving-average window applied to each signal before alignment.
    scaling :
        Amplitude-scaling method. ``"none"`` (default) only shifts and
        resamples, preserving the y axes, labels, and units. ``"feature"``
        min-max scales each signal to ``[0, 1]``; ``"zscore"`` standardizes
        each to zero mean / unit std; ``"trigger"`` performs a quantitative
        two-point calibration that maps the time trace onto the KPFM target
        scale via the laser triggers. ``"feature"``/``"zscore"`` statistics
        are taken over the laser-on (trigger-bounded) window.
    trigger_window :
        Half-width (in samples) of the median filter applied to each trigger
        anchor when ``scaling="trigger"``. ``<= 1`` uses the single trigger
        sample; larger values add robustness against noise at the triggers.
    """

    def __init__(
        self,
        kpfm: Signal,
        timetrace: Signal,
        *,
        kpfm_triggers: _TriggerSpec,
        timetrace_triggers: _TriggerSpec,
        n_points: int | None = None,
        kpfm_smoothing: int = 1,
        timetrace_smoothing: int = 1,
        scaling: str = "none",
        trigger_window: int = 5,
    ) -> None:
        k_t, _ = _validate_signal(kpfm, "kpfm")
        t_t, _ = _validate_signal(timetrace, "timetrace")
        _validated_trigger_times(kpfm_triggers, "kpfm_triggers")
        _validated_trigger_times(timetrace_triggers, "timetrace_triggers")
        for name, value in (
            ("kpfm_smoothing", kpfm_smoothing),
            ("timetrace_smoothing", timetrace_smoothing),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if n_points is not None and (
            isinstance(n_points, bool)
            or not isinstance(n_points, (int, np.integer))
            or n_points < 2
        ):
            raise ValueError("n_points must be an integer of at least 2.")

        self.kpfm = kpfm
        self.timetrace = timetrace
        self.kpfm_triggers = kpfm_triggers
        self.timetrace_triggers = timetrace_triggers

        self._k_smooth = kpfm_smoothing
        self._t_smooth = timetrace_smoothing
        self.scaling = scaling
        self._trigger_window = trigger_window

        # The input axes have already been checked here so failures occur
        # before smoothing, interpolation, or trigger-index access.
        self._input_axes = (k_t, t_t)

        self._align(n_points)

    # ------------------------------------------------------------------ align
    def _align(self, n_points: int | None) -> None:
        k_t = self._input_axes[0]
        k_y = smooth(np.asarray(self.kpfm["y"], dtype=float), self._k_smooth)
        t_t = self._input_axes[1]
        t_y = smooth(np.asarray(self.timetrace["y"], dtype=float), self._t_smooth)

        k_left, k_right = _validated_trigger_times(
            self.kpfm_triggers, "kpfm_triggers"
        )
        t_left, t_right = _validated_trigger_times(
            self.timetrace_triggers, "timetrace_triggers"
        )
        self._single_trigger = k_left == k_right

        # 1) shift the time trace so its left trigger meets the KPFM left trigger.
        offset = t_left - k_left
        t_t_shift = t_t - offset

        # Validate overlap before scaling so invalid inputs fail consistently,
        # independent of the selected scaling mode.
        t0 = max(k_t.min(), t_t_shift.min())
        t1 = min(k_t.max(), t_t_shift.max())
        if t1 <= t0:
            raise ValueError(
                "kpfm and shifted timetrace have no overlapping time span."
            )

        # 2) amplitude-scale both signals according to the chosen method.
        (
            self.kpfm_scale,
            self.kpfm_shift,
            self.scale,
            self.shift,
        ) = self._compute_scaling(k_y, k_t, t_y, t_t_shift, k_left, k_right)
        k_x_unit = self.kpfm.units.get("x", "")
        if self.scaling == "none":
            self._info = {
                "k_label": self.kpfm.label,
                "t_label": self.timetrace.label,
                "k_unit": self.kpfm.units.get("y", ""),
                "t_unit": self.timetrace.units.get("y", ""),
                "x_unit": k_x_unit,
            }
        elif self.scaling == "trigger":
            target_unit = self.kpfm.units.get("y", "")
            self._info = {
                **_SCALING_INFO["trigger"],
                "k_label": self.kpfm.label,
                "t_label": self.timetrace.label,
                "k_unit": target_unit,
                "t_unit": target_unit,
                "x_unit": k_x_unit,
            }
        else:
            self._info = {**_SCALING_INFO[self.scaling], "x_unit": k_x_unit}
        k_y_scaled = self.kpfm_scale * k_y + self.kpfm_shift
        t_y_scaled = self.scale * t_y + self.shift

        # 3) build a shared uniform grid over the overlapping time span.
        if n_points is None:
            k_res = (k_t.max() - k_t.min()) / max(len(k_t) - 1, 1)
            t_res = (t_t_shift.max() - t_t_shift.min()) / max(len(t_t_shift) - 1, 1)
            n_points = int(max((t1 - t0) / min(k_res, t_res), 2)) + 1
        grid = np.linspace(t0, t1, n_points)

        self.grid = grid
        self._k_y_aligned = k_y_scaled
        self._k_t_aligned = k_t
        self._t_y_aligned = t_y_scaled
        self._t_t_aligned = t_t_shift
        self._k_grid = resample(k_t, k_y_scaled, grid)
        self._t_grid = resample(t_t_shift, t_y_scaled, grid)

        # Store trigger points in the shared loader shape. Their indices and
        # times refer to the common grid, so the point invariant always holds.
        left_index = int(np.argmin(np.abs(grid - k_left)))
        right_index = int(np.argmin(np.abs(grid - k_right)))
        self.triggers: TriggerPair = {
            "left": {"index": left_index, "time": float(grid[left_index])},
            "right": {"index": right_index, "time": float(grid[right_index])},
        }

    # ---------------------------------------------------------------- scaling
    def _compute_scaling(
        self,
        k_y: np.ndarray,
        k_t: np.ndarray,
        t_y: np.ndarray,
        t_t_shift: np.ndarray,
        k_left: float,
        k_right: float,
    ) -> tuple[float, float, float, float]:
        """Return ``(kpfm_scale, kpfm_shift, tt_scale, tt_shift)`` for the
        selected :attr:`scaling` method.

        For ``"feature"``/``"zscore"`` each signal gets its own affine transform,
        with statistics taken over the laser-on (trigger-bounded) window so the
        laser-off baseline cannot dominate. With a single trigger (left equals
        right) no window exists and the statistics fall back to the full
        overlap. For ``"trigger"`` the KPFM signal is
        left untouched and the time trace is mapped onto the KPFM scale using the
        two (optionally median-filtered) trigger anchors. For ``"none"`` both
        signals are passed through unscaled (identity transform).
        """
        method = self.scaling
        if method == "none":
            return 1.0, 0.0, 1.0, 0.0
        if method == "trigger":
            ki, ti = self.kpfm_triggers, self.timetrace_triggers
            if not (isinstance(ki, dict) and isinstance(ti, dict)):
                raise ValueError(
                    "scaling='trigger' needs both triggers as dicts with 'left' "
                    "and 'right' points; a single float trigger only supports "
                    "scaling='feature' or 'zscore'."
                )
            indices = (
                ("kpfm", int(ki["left"]["index"]), len(k_y)),
                ("kpfm", int(ki["right"]["index"]), len(k_y)),
                ("timetrace", int(ti["left"]["index"]), len(t_y)),
                ("timetrace", int(ti["right"]["index"]), len(t_y)),
            )
            if any(index < 0 or index >= length for _, index, length in indices):
                raise ValueError(
                    "trigger indices must refer to samples in their Signal."
                )
            w = self._trigger_window
            k_y1 = _robust_trigger_value(k_y, int(ki["left"]["index"]), w)
            k_y2 = _robust_trigger_value(k_y, int(ki["right"]["index"]), w)
            t_y1 = _robust_trigger_value(t_y, int(ti["left"]["index"]), w)
            t_y2 = _robust_trigger_value(t_y, int(ti["right"]["index"]), w)
            if not np.all(np.isfinite([k_y1, k_y2, t_y1, t_y2])):
                raise ValueError(
                    "trigger scaling requires finite amplitudes at all anchors."
                )
            denom = t_y2 - t_y1
            # Use the magnitude only: matching both endpoints' signs would flip
            # the time trace whenever KPFM and PL move oppositely between the
            # triggers. abs() keeps the PL trace's natural orientation (it rises
            # while the light is on) and anchors it to the left trigger.
            scale = abs((k_y2 - k_y1) / denom) if denom != 0 else 1.0
            if denom == 0 or scale == 0 or not np.isfinite(scale): # This ensures that if there is only one trigger point, meaning left=right, it will work like method="none"
                warnings.warn(
                    "scaling='trigger' has degenerate anchors (identical "
                    "trigger amplitudes, e.g. a single trigger point); "
                    "falling back to unit scale like scaling='none'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                scale = 1.0
            shift = k_y1 - scale * t_y1
            return 1.0, 0.0, float(scale), float(shift)

        if self._single_trigger:
            # Single trigger (left == right): no laser-on window is defined, so
            # take the statistics over the full overlap -- mirroring the
            # `_window_mask` fallback used by the statistical helpers.
            k_mask = np.isfinite(k_y)
            t_mask = np.isfinite(t_y)
        else:
            k_mask = (k_t >= k_left) & (k_t <= k_right) & np.isfinite(k_y)
            t_mask = (t_t_shift >= k_left) & (t_t_shift <= k_right) & np.isfinite(t_y)
        if not np.any(k_mask) or not np.any(t_mask):
            raise ValueError(
                f"scaling={method!r} requires finite samples in both trigger windows."
            )
        if method == "feature":
            ks, ksh = feature_scale_params(k_y, k_mask)
            ts, tsh = feature_scale_params(t_y, t_mask)
        elif method == "zscore":
            ks, ksh = zscore_params(k_y, k_mask)
            ts, tsh = zscore_params(t_y, t_mask)
        else:
            raise ValueError(
                f"Unknown scaling method: {method!r} "
                "(expected 'none', 'feature', 'zscore', or 'trigger')"
            )
        return ks, ksh, ts, tsh

    # ------------------------------------------------------------------ output
    def kpfm_signal(self, on_grid: bool = True) -> Signal:
        """Aligned KPFM signal -- on the shared grid (default) or native axis."""
        info = self._info
        meta = {"triggers": self.triggers, "scaling": self.scaling,
                "scale": self.kpfm_scale, "shift": self.kpfm_shift,
                "source": {"label": self.kpfm.label,
                           "units": dict(self.kpfm.units),
                           "meta": dict(self.kpfm.meta)}}
        if on_grid:
            return Signal(self.grid, self._k_grid, label=info["k_label"],
                          units={"x": info["x_unit"], "y": info["k_unit"]},
                          meta={**meta, "smoothing": self._k_smooth})
        return Signal(self._k_t_aligned, self._k_y_aligned, label=info["k_label"],
                      units={"x": info["x_unit"], "y": info["k_unit"]}, meta=meta)

    def timetrace_signal(self, on_grid: bool = True) -> Signal:
        """Aligned, amplitude-scaled time trace -- shared grid or native axis."""
        info = self._info
        meta = {"scale": self.scale, "shift": self.shift, "scaling": self.scaling,
                "triggers": self.triggers, "smoothing": self._t_smooth,
                "source": {"label": self.timetrace.label,
                           "units": dict(self.timetrace.units),
                           "meta": dict(self.timetrace.meta)}}
        if on_grid:
            return Signal(self.grid, self._t_grid, label=info["t_label"],
                          units={"x": info["x_unit"], "y": info["t_unit"]}, meta=meta)
        return Signal(self._t_t_aligned, self._t_y_aligned,
                      label=info["t_label"],
                      units={"x": info["x_unit"], "y": info["t_unit"]}, meta=meta)

    def to_dict(self) -> dict[str, Any]:
        """Return both aligned signals (on the shared grid) plus metadata."""
        return {
            "grid": self.grid,
            "kpfm": self.kpfm_signal().to_dict(),
            "timetrace": self.timetrace_signal().to_dict(),
            "scaling": self.scaling,
            "scale": self.scale,
            "shift": self.shift,
            "kpfm_scale": self.kpfm_scale,
            "kpfm_shift": self.kpfm_shift,
            "triggers": self.triggers,
        }

    def kpfm_derivative(self, smoothing: int = 1, on_grid: bool = True) -> Signal:
        """Time derivative of the aligned KPFM signal as a :class:`Signal`."""
        s = self.kpfm_signal(on_grid)
        dy = derivative(s["x"], s["y"], smoothing)
        return Signal(s["x"], dy, label=r"$d\Phi/dt$",
                      units={"x": self._info["x_unit"],
                             "y": f"{self._info['k_unit']}/{self._info['x_unit']}"},
                      meta={"triggers": self.triggers})

    # ------------------------------------------------------------------ stats
    def _window_mask(self, x: np.ndarray) -> np.ndarray:
        # Single-trigger case (left == right): no laser-on window is defined, so
        # fall back to the full overlap.
        if self._single_trigger:
            return np.ones_like(x, bool)
        return (
            (x >= self.triggers["left"]["time"])
            & (x <= self.triggers["right"]["time"])
        )

    def _statistical_samples(
        self, within_triggers: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return paired finite samples, rejecting underspecified statistics."""
        mask = (
            self._window_mask(self.grid)
            if within_triggers
            else np.ones_like(self.grid, dtype=bool)
        )
        valid = mask & np.isfinite(self._k_grid) & np.isfinite(self._t_grid)
        k, t = self._k_grid[valid], self._t_grid[valid]
        if k.size < 2:
            raise ValueError("statistics require at least two paired finite samples.")
        if np.ptp(k) == 0 or np.ptp(t) == 0:
            raise ValueError("statistics require non-constant paired signals.")
        return k, t

    def pearson(self, within_triggers: bool = True) -> dict[str, float]:
        """Pearson correlation between the two aligned signals on the grid.

        Parameters
        ----------
        within_triggers :
            Restrict to the laser-on window between the two triggers.
        """
        k, t = self._statistical_samples(within_triggers)
        r = stats.pearsonr(k, t)
        return {"statistic": float(r.statistic), "pvalue": float(r.pvalue)}

    def cross_correlation(self, within_triggers: bool = True,
                          normalize: bool = True) -> dict[str, np.ndarray | float]:
        """Cross-correlation of the two aligned signals and the best lag.

        Returns ``{"lag": <lags in s>, "correlation": <array>, "best_lag": <s>,
        "best_correlation": <float>}``.
        Positive ``best_lag`` means the time trace leads the KPFM signal
        (or equivalently, the KPFM signal lags the time trace).
        Negative ``best_lag`` means the time trace lags the KPFM signal.
        """
        k, t = self._statistical_samples(within_triggers)
        k = k - k.mean()
        t = t - t.mean()

        corr = np.correlate(k, t, mode="full")
        if normalize:
            denom = np.sqrt(np.sum(k ** 2) * np.sum(t ** 2))
            if denom > 0:
                corr = corr / denom

        dt = self.grid[1] - self.grid[0] if len(self.grid) > 1 else 1.0
        lags = np.arange(-len(t) + 1, len(k)) * dt
        idx = np.argmax(np.abs(corr))
        best_lag = float(lags[idx])
        best_corr = float(corr[idx])
        return {"lag": lags, "correlation": corr, "best_lag": best_lag, "best_correlation": best_corr}
