"""Align a PL time trace with a KPFM signal on a shared time grid.

The :class:`Match` class takes two :class:`~kpfm_pl.core.Signal` traces plus the
laser triggers of each, then:

1. shifts the time trace so one chosen trigger coincides with its KPFM
   counterpart,
2. resamples both signals onto a shared, uniform time grid that spans every
   data point of both signals,
3. maps every KPFM trigger onto that grid.

Use :meth:`Match.extract` to cut out the region of interest, either between two
triggers or between two times. Everything is returned as :class:`Signal`/dict
objects; there is no plotting, amplitude scaling or statistics -- do those in
post.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import Signal
from .utils import TriggerPoint, resample, smooth, trigger_points

__all__ = ["Match"]


_TriggerSpec = list[TriggerPoint] | TriggerPoint | np.ndarray | float
_Bound = TriggerPoint | float | None


def _trigger_times(triggers: _TriggerSpec, name: str) -> np.ndarray:
    """Trigger times from trigger point(s) or plain time(s) in seconds."""
    if isinstance(triggers, dict):
        triggers = [triggers]
    try:
        times = np.array([
            float(point["time"]) if isinstance(point, dict) else float(point)
            for point in np.atleast_1d(np.asarray(triggers, dtype=object))
        ])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be trigger point(s) or time(s) in seconds."
        ) from exc
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError(f"{name} must contain at least one finite trigger time.")
    return times


def _bound_time(bound: _Bound, default: float, name: str) -> float:
    """Time of an :meth:`Match.extract` bound (a time or a trigger point)."""
    if bound is None:
        return default
    try:
        t = float(bound["time"]) if isinstance(bound, dict) else float(bound)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a time or a trigger point.") from exc
    if not np.isfinite(t):
        raise ValueError(f"{name} must be finite.")
    return t


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


class Match:
    """Align a KPFM signal and a PL time trace on a shared time grid.

    Parameters
    ----------
    kpfm, timetrace :
        The two :class:`Signal` traces to align.
    kpfm_triggers, timetrace_triggers :
        A list of trigger points (``[{"index", "time"}, ...]``) as returned by
        :meth:`~kpfm_pl.zurich.Zurich.laser_triggers` and
        :meth:`~kpfm_pl.ptu.PTU.laser_triggers`, a single trigger point, or
        plain time(s) in seconds (e.g. the exact marker times from
        :meth:`~kpfm_pl.ptu.PTU.get_markers`).
    align :
        Position of the trigger, in both trigger lists, used to align the two
        signals. Default ``0`` (the first trigger); negative values count from
        the end. The time trace is shifted by
        ``timetrace_triggers[align]["time"] - kpfm_triggers[align]["time"]``.
    dt :
        Spacing of the shared grid in seconds. ``None`` uses the finer of the
        two native resolutions.
    kpfm_smoothing, timetrace_smoothing :
        Optional moving-average window applied to each signal before alignment.

    Attributes
    ----------
    offset : float
        Time shift subtracted from the time-trace axis.
    grid : np.ndarray
        Shared uniform time grid. It spans every sample of both signals and
        passes exactly through the aligned trigger.
    triggers : list[TriggerPoint]
        Every KPFM trigger mapped onto :attr:`grid`. After alignment the
        time-trace triggers live in the same frame, so one list serves both.
    """

    def __init__(
        self, kpfm: Signal, timetrace: Signal, *,
        kpfm_triggers: _TriggerSpec, timetrace_triggers: _TriggerSpec,
        align: int = 0, dt: float | None = None,
        kpfm_smoothing: int = 1, timetrace_smoothing: int = 1,) -> None:

        k_t, k_y = _validate_signal(kpfm, "kpfm")
        t_t, t_y = _validate_signal(timetrace, "timetrace")
        k_times = _trigger_times(kpfm_triggers, "kpfm_triggers")
        t_times = _trigger_times(timetrace_triggers, "timetrace_triggers")

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
        n_common = min(k_times.size, t_times.size)
        if (
            isinstance(align, bool)
            or not isinstance(align, (int, np.integer))
            or not -n_common <= align < n_common
        ):
            raise ValueError(
                f"align must be an integer index into both trigger lists "
                f"({k_times.size} KPFM, {t_times.size} time-trace triggers)."
            )
        if dt is not None and (
            isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0
        ):
            raise ValueError("dt must be a positive finite number.")

        self.kpfm = kpfm
        self.timetrace = timetrace
        self._k_smooth = kpfm_smoothing
        self._t_smooth = timetrace_smoothing

        # 1) shift the time trace so its aligned trigger meets the KPFM one.
        anchor = float(k_times[align])
        self.offset = float(t_times[align] - anchor)
        t_t_shift = t_t - self.offset

        lo, hi = min(k_t[0], t_t_shift[0]), max(k_t[-1], t_t_shift[-1])
        self._overlap = (max(k_t[0], t_t_shift[0]), min(k_t[-1], t_t_shift[-1]))
        if self._overlap[1] <= self._overlap[0]:
            raise ValueError("kpfm and shifted timetrace have no overlapping time span.")

        # 2) build a uniform grid over all samples of both signals, anchored on
        # the aligned trigger so that trigger is exactly one grid sample.
        if dt is None:
            k_res = (k_t[-1] - k_t[0]) / (len(k_t) - 1)
            t_res = (t_t[-1] - t_t[0]) / (len(t_t) - 1)
            dt = min(k_res, t_res)
        self.dt = float(dt)
        first = int(np.ceil((lo - anchor) / self.dt - 1e-9))
        last = int(np.floor((hi - anchor) / self.dt + 1e-9))
        self.grid = anchor + self.dt * np.arange(first, last + 1)

        self._k_t, self._k_y = k_t, smooth(k_y, kpfm_smoothing)
        self._t_t, self._t_y = t_t_shift, smooth(t_y, timetrace_smoothing)
        self._k_grid = resample(self._k_t, self._k_y, self.grid)
        self._t_grid = resample(self._t_t, self._t_y, self.grid)

        # 3) every KPFM trigger on the shared grid (grid[index] == time).
        self.triggers: list[TriggerPoint] = trigger_points(k_times, self.grid)

    # ------------------------------------------------------------------ output
    def _triggers_on(self, x: np.ndarray) -> list[TriggerPoint]:
        """The triggers that fall within ``x``, re-indexed onto ``x``."""
        times = np.array([point["time"] for point in self.triggers])
        times = times[(times >= x[0]) & (times <= x[-1])]
        return trigger_points(times, x)

    def _signal(self, source: Signal, x: np.ndarray, y: np.ndarray, *,
                smoothing: int, time_shift: float) -> Signal:
        return Signal(x, y, label=source.label,
                      units={"x": self.kpfm.units.get("x", ""),
                             "y": source.units.get("y", "")},
                      meta={"triggers": self._triggers_on(x),
                            "time_shift": time_shift, "smoothing": smoothing,
                            "source": {"label": source.label,
                                       "units": dict(source.units),
                                       "meta": dict(source.meta)}})

    def kpfm_signal(self, on_grid: bool = True) -> Signal:
        """Aligned KPFM signal -- on the shared grid (default) or native axis.

        ``meta["triggers"]`` holds the triggers indexed onto the returned axis.
        On the grid, samples outside the KPFM time span are ``NaN``.
        """
        x, y = (self.grid, self._k_grid) if on_grid else (self._k_t, self._k_y)
        return self._signal(self.kpfm, x, y, smoothing=self._k_smooth, time_shift=0.0)

    def timetrace_signal(self, on_grid: bool = True) -> Signal:
        """Aligned time trace -- on the shared grid (default) or native axis.

        The native axis is shifted by :attr:`offset`. ``meta["triggers"]``
        holds the triggers indexed onto the returned axis. On the grid,
        samples outside the time-trace span are ``NaN``.
        """
        x, y = (self.grid, self._t_grid) if on_grid else (self._t_t, self._t_y)
        return self._signal(self.timetrace, x, y, smoothing=self._t_smooth,
                            time_shift=self.offset)

    def extract(self, start: _Bound = None, stop: _Bound = None) -> dict[str, Signal]:
        """Cut both aligned signals to ``[start, stop]`` on the shared grid.

        Parameters
        ----------
        start, stop :
            Inclusive bounds, either a time on the grid in seconds or a trigger
            point such as ``m.triggers[0]``. ``None`` defaults to the start/end
            of the region where both signals have data.

        Returns
        -------
        dict
            ``{"kpfm": Signal, "timetrace": Signal}`` on the cropped grid. The
            triggers inside the window are stored in each signal's
            ``meta["triggers"]``, indexed onto the cropped grid. Samples
            outside a signal's own time span are ``NaN``.

        Examples
        --------
        >>> m.extract(m.triggers[0], m.triggers[-1])   # between two triggers
        >>> m.extract(10.0, 300.0)                     # between two times
        """
        t0 = _bound_time(start, self._overlap[0], "start")
        t1 = _bound_time(stop, self._overlap[1], "stop")
        if t1 < t0:
            raise ValueError("stop must not precede start.")
        # Tolerance so bounds given as times hit grid samples despite rounding.
        eps = 1e-9 * self.dt
        index = np.flatnonzero((self.grid >= t0 - eps) & (self.grid <= t1 + eps))
        if index.size == 0:
            raise ValueError(f"no grid samples between {t0} s and {t1} s.")
        window = slice(index[0], index[-1] + 1)
        x = self.grid[window]
        return {
            "kpfm": self._signal(self.kpfm, x, self._k_grid[window],
                                 smoothing=self._k_smooth, time_shift=0.0),
            "timetrace": self._signal(self.timetrace, x, self._t_grid[window],
                                      smoothing=self._t_smooth,
                                      time_shift=self.offset),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return both aligned signals (on the shared grid) plus metadata."""
        return {
            "grid": self.grid,
            "kpfm": self.kpfm_signal().to_dict(),
            "timetrace": self.timetrace_signal().to_dict(),
            "offset": self.offset,
            "triggers": self.triggers,
        }
