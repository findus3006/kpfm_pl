"""Align a PL time trace with a KPFM signal for joint analysis.

The :class:`Match` class takes two :class:`~kpfm_pl.core.Signal` traces plus the
laser triggers of each, then:

1. shifts the time axes so the *left* triggers coincide,
2. linearly rescales the time-trace amplitude onto the KPFM scale using the two
   trigger points,
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
from .utils import derivative, resample, smooth

__all__ = ["Match"]


def _trigger_times(triggers: dict[str, dict[str, float]]) -> tuple[float, float]:
    return triggers["left"]["time"], triggers["right"]["time"]


class Match:
    """Align and co-scale a KPFM signal and a PL time trace.

    Parameters
    ----------
    kpfm, timetrace :
        The two :class:`Signal` traces to align.
    kpfm_triggers, timetrace_triggers :
        Trigger dicts (``{"left": {"index", "time"}, "right": ...}``) as returned
        by :meth:`~kpfm_pl.zurich.Zurich.laser_triggers` and
        :meth:`~kpfm_pl.ptu.PTU.laser_triggers`.
    n_points :
        Number of samples in the shared resampled grid. ``None`` uses the finer
        of the two native resolutions over the overlap.
    kpfm_smoothing, timetrace_smoothing :
        Optional moving-average window applied to each signal before alignment.
    """

    def __init__(
        self,
        kpfm: Signal,
        timetrace: Signal,
        *,
        kpfm_triggers: dict[str, dict[str, float]],
        timetrace_triggers: dict[str, dict[str, float]],
        n_points: int | None = None,
        kpfm_smoothing: int = 1,
        timetrace_smoothing: int = 1,
    ) -> None:
        self.kpfm = kpfm
        self.timetrace = timetrace
        self.kpfm_triggers = kpfm_triggers
        self.timetrace_triggers = timetrace_triggers

        self._k_smooth = kpfm_smoothing
        self._t_smooth = timetrace_smoothing

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

        # 2) amplitude-rescale the time trace onto the KPFM scale via trigger pts.
        ki = self.kpfm_triggers
        ti = self.timetrace_triggers
        k_y1, k_y2 = k_y[ki["left"]["index"]], k_y[ki["right"]["index"]]
        t_y1, t_y2 = t_y[ti["left"]["index"]], t_y[ti["right"]["index"]]
        denom = t_y2 - t_y1
        scale = abs((k_y2 - k_y1) / denom) if denom != 0 else 1.0
        if scale == 0 or not np.isfinite(scale):
            scale = 1.0
        shift = k_y1 - scale * t_y1
        t_y_scaled = scale * t_y + shift
        self.scale, self.shift = float(scale), float(shift)

        # 3) build a shared uniform grid over the overlapping time span.
        t0 = max(k_t.min(), t_t_shift.min())
        t1 = min(k_t.max(), t_t_shift.max())
        if n_points is None:
            k_res = (k_t.max() - k_t.min()) / max(len(k_t) - 1, 1)
            t_res = (t_t_shift.max() - t_t_shift.min()) / max(len(t_t_shift) - 1, 1)
            n_points = int(max((t1 - t0) / min(k_res, t_res), 2)) + 1
        grid = np.linspace(t0, t1, n_points)

        self.grid = grid
        self._k_y_aligned = k_y
        self._k_t_aligned = k_t
        self._t_y_aligned = t_y_scaled
        self._t_t_aligned = t_t_shift
        self._k_grid = resample(k_t, k_y, grid)
        self._t_grid = resample(t_t_shift, t_y_scaled, grid)

        # Trigger times on the common (KPFM) reference frame.
        self.triggers = {"left": k_left, "right": k_right}

    # ------------------------------------------------------------------ output
    def kpfm_signal(self, on_grid: bool = True) -> Signal:
        """Aligned KPFM signal -- on the shared grid (default) or native axis."""
        if on_grid:
            return Signal(self.grid, self._k_grid, label="KPFM (aligned)",
                          units={"x": "s", "y": "V"}, meta={"triggers": self.triggers})
        return Signal(self._k_t_aligned, self._k_y_aligned, label="KPFM (aligned)",
                      units={"x": "s", "y": "V"}, meta={"triggers": self.triggers})

    def timetrace_signal(self, on_grid: bool = True) -> Signal:
        """Aligned, amplitude-scaled time trace -- shared grid or native axis."""
        meta = {"scale": self.scale, "shift": self.shift, "triggers": self.triggers}
        if on_grid:
            return Signal(self.grid, self._t_grid, label="time trace (aligned, scaled)",
                          units={"x": "s", "y": "counts (scaled)"}, meta=meta)
        return Signal(self._t_t_aligned, self._t_y_aligned,
                      label="time trace (aligned, scaled)",
                      units={"x": "s", "y": "counts (scaled)"}, meta=meta)

    def to_dict(self) -> dict[str, Any]:
        """Return both aligned signals (on the shared grid) plus metadata."""
        return {
            "grid": self.grid,
            "kpfm": self.kpfm_signal().to_dict(),
            "timetrace": self.timetrace_signal().to_dict(),
            "scale": self.scale,
            "shift": self.shift,
            "triggers": self.triggers,
        }

    def kpfm_derivative(self, smoothing: int = 1, on_grid: bool = True) -> Signal:
        """Time derivative of the aligned KPFM signal as a :class:`Signal`."""
        s = self.kpfm_signal(on_grid)
        dy = derivative(s["x"], s["y"], smoothing)
        return Signal(s["x"], dy, label=r"$d\Phi/dt$",
                      units={"x": "s", "y": "V/s"}, meta={"triggers": self.triggers})

    # ------------------------------------------------------------------ stats
    def _window_mask(self, x: np.ndarray) -> np.ndarray:
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
