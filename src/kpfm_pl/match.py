"""Align a PL time trace with a KPFM signal for joint analysis.

The :class:`Match` class takes two :class:`~kpfm_pl.core.Signal` traces plus the
laser triggers of each, then:

1. shifts the time axes so the *left* triggers coincide,
2. amplitude-scales the two signals with a selectable method (``"feature"``
   min-max to ``[0, 1]`` by default, ``"zscore"`` standardization, or the
   quantitative two-point ``"trigger"`` calibration onto the KPFM scale),
3. resamples both onto a shared, uniform time grid.

Everything is returned as dicts/:class:`Signal` objects; there is no plotting.
Statistical helpers (Pearson correlation, cross-correlation/lag) operate on the
trigger-bounded overlap region.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .core import Signal
from .utils import (
    derivative,
    feature_scale_params,
    resample,
    smooth,
    zscore_params,
)

__all__ = ["Match"]


TriggerSpec = dict[str, dict[str, float]] | float


def _trigger_times(triggers: TriggerSpec) -> tuple[float, float]:
    """Left/right trigger times.

    Accepts either the full dict (``{"left": {"index", "time"}, "right": ...}``)
    or a single ``float`` time, in which case the right trigger equals the left.
    """
    if isinstance(triggers, dict):
        return triggers["left"]["time"], triggers["right"]["time"]
    t = float(triggers)
    return t, t


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
    "none": {
        "k_label": "KPFM (aligned)",
        "t_label": "time trace (aligned)",
        "k_unit": "V",
        "t_unit": "counts",
    },
    "feature": {
        "k_label": "KPFM (aligned, normalized)",
        "t_label": "time trace (aligned, normalized)",
        "k_unit": "norm. [0, 1]",
        "t_unit": "norm. [0, 1]",
    },
    "zscore": {
        "k_label": "KPFM (aligned, z-score)",
        "t_label": "time trace (aligned, z-score)",
        "k_unit": "z-score",
        "t_unit": "z-score",
    },
    "trigger": {
        "k_label": "KPFM (aligned)",
        "t_label": "time trace (aligned, scaled)",
        "k_unit": "V",
        "t_unit": "V (scaled)",
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
        Amplitude-scaling method. ``"feature"`` (default) min-max scales each
        signal to ``[0, 1]``; ``"zscore"`` standardizes each to zero mean / unit
        std; ``"trigger"`` performs the quantitative two-point calibration that
        maps the time trace onto the KPFM voltage scale via the laser triggers;
        ``"none"`` only shifts and resamples (the y axes are left untouched, e.g.
        to scale manually at plot time). ``"feature"``/``"zscore"`` statistics are
        taken over the laser-on (trigger-bounded) window.
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
        kpfm_triggers: TriggerSpec,
        timetrace_triggers: TriggerSpec,
        n_points: int | None = None,
        kpfm_smoothing: int = 1,
        timetrace_smoothing: int = 1,
        scaling: str = "none",
        trigger_window: int = 5,
    ) -> None:
        self.kpfm = kpfm
        self.timetrace = timetrace
        self.kpfm_triggers = kpfm_triggers
        self.timetrace_triggers = timetrace_triggers

        self._k_smooth = kpfm_smoothing
        self._t_smooth = timetrace_smoothing
        self.scaling = scaling
        self._trigger_window = trigger_window

        self._align(n_points)

    # ------------------------------------------------------------------ align
    def _align(self, n_points: int | None) -> None:
        k_t = np.asarray(self.kpfm["x"], dtype=float)
        k_y = smooth(np.asarray(self.kpfm["y"], dtype=float), self._k_smooth)
        t_t = np.asarray(self.timetrace["x"], dtype=float)
        t_y = smooth(np.asarray(self.timetrace["y"], dtype=float), self._t_smooth)

        k_left, k_right = _trigger_times(self.kpfm_triggers)
        t_left, t_right = _trigger_times(self.timetrace_triggers)

        # 1) shift the time trace so its left trigger meets the KPFM left trigger.
        offset = t_left - k_left
        t_t_shift = t_t - offset

        # 2) amplitude-scale both signals according to the chosen method.
        (self.kpfm_scale, self.kpfm_shift, self.scale, self.shift) = self._compute_scaling(       
            k_y, k_t, t_y, t_t_shift, k_left, k_right)
        self._info = _SCALING_INFO[self.scaling]
        k_y_scaled = self.kpfm_scale * k_y + self.kpfm_shift
        t_y_scaled = self.scale * t_y + self.shift

        # 3) build a shared uniform grid over the overlapping time span.
        t0 = max(k_t.min(), t_t_shift.min())
        t1 = min(k_t.max(), t_t_shift.max())
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

        # Trigger times on the common (KPFM) reference frame.
        self.triggers = {"left": k_left, "right": k_right}

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
        laser-off baseline cannot dominate. For ``"trigger"`` the KPFM signal is
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
            w = self._trigger_window
            k_y1 = _robust_trigger_value(k_y, int(ki["left"]["index"]), w)
            k_y2 = _robust_trigger_value(k_y, int(ki["right"]["index"]), w)
            t_y1 = _robust_trigger_value(t_y, int(ti["left"]["index"]), w)
            t_y2 = _robust_trigger_value(t_y, int(ti["right"]["index"]), w)
            denom = t_y2 - t_y1
            # Use the magnitude only: matching both endpoints' signs would flip
            # the time trace whenever KPFM and PL move oppositely between the
            # triggers. abs() keeps the PL trace's natural orientation (it rises
            # while the light is on) and anchors it to the left trigger.
            scale = abs((k_y2 - k_y1) / denom) if denom != 0 else 1.0
            if scale == 0 or not np.isfinite(scale):
                scale = 1.0
            shift = k_y1 - scale * t_y1
            return 1.0, 0.0, float(scale), float(shift)

        k_mask = (k_t >= k_left) & (k_t <= k_right)
        t_mask = (t_t_shift >= k_left) & (t_t_shift <= k_right)
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
                "scale": self.kpfm_scale, "shift": self.kpfm_shift}
        if on_grid:
            return Signal(self.grid, self._k_grid, label=info["k_label"],
                          units={"x": "s", "y": info["k_unit"]},
                          meta={**meta, "smoothing": self._k_smooth})
        return Signal(self._k_t_aligned, self._k_y_aligned, label=info["k_label"],
                      units={"x": "s", "y": info["k_unit"]}, meta=meta)

    def timetrace_signal(self, on_grid: bool = True) -> Signal:
        """Aligned, amplitude-scaled time trace -- shared grid or native axis."""
        info = self._info
        meta = {"scale": self.scale, "shift": self.shift, "scaling": self.scaling,
                "triggers": self.triggers, "smoothing": self._t_smooth}
        if on_grid:
            return Signal(self.grid, self._t_grid, label=info["t_label"],
                          units={"x": "s", "y": info["t_unit"]}, meta=meta)
        return Signal(self._t_t_aligned, self._t_y_aligned,
                      label=info["t_label"],
                      units={"x": "s", "y": info["t_unit"]}, meta=meta)

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
                      units={"x": "s", "y": f"{self._info['k_unit']}/s"},
                      meta={"triggers": self.triggers})

    # ------------------------------------------------------------------ stats
    def _window_mask(self, x: np.ndarray) -> np.ndarray:
        # Single-trigger case (left == right): no laser-on window is defined, so
        # fall back to the full overlap.
        if self.triggers["left"] == self.triggers["right"]:
            return np.ones_like(x, bool)
        return (x >= self.triggers["left"]) & (x <= self.triggers["right"])

    def pearson(self, within_triggers: bool = True) -> dict[str, float]:
        """Pearson correlation between the two aligned signals on the grid.

        Parameters
        ----------
        within_triggers :
            Restrict to the laser-on window between the two triggers.
        """
        k, t = self._k_grid, self._t_grid
        mask = self._window_mask(self.grid) if within_triggers else np.ones_like(self.grid, bool)
        valid = mask & np.isfinite(k) & np.isfinite(t)
        r = stats.pearsonr(k[valid], t[valid])
        return {"statistic": float(r.statistic), "pvalue": float(r.pvalue)}

    def cross_correlation(self, within_triggers: bool = True,
                          normalize: bool = True) -> dict[str, np.ndarray | float]:
        """Cross-correlation of the two aligned signals and the best lag.

        Returns ``{"lag": <lags in s>, "correlation": <array>, "best_lag": <s>}``.
        Positive ``best_lag`` means the time trace lags the KPFM signal.
        """
        mask = self._window_mask(self.grid) if within_triggers else np.ones_like(self.grid, bool)
        k = np.nan_to_num(self._k_grid[mask])
        t = np.nan_to_num(self._t_grid[mask])
        k = k - k.mean()
        t = t - t.mean()

        corr = np.correlate(k, t, mode="full")
        if normalize:
            denom = np.sqrt(np.sum(k ** 2) * np.sum(t ** 2))
            if denom > 0:
                corr = corr / denom

        dt = self.grid[1] - self.grid[0] if len(self.grid) > 1 else 1.0
        lags = np.arange(-len(t) + 1, len(k)) * dt
        best_lag = float(lags[int(np.argmax(corr))])
        return {"lag": lags, "correlation": corr, "best_lag": best_lag}
