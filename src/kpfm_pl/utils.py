"""Reusable, dependency-light signal-processing utilities.

These functions are pure (no I/O, no plotting) so they are easy to test and to
reuse across the loaders and the :class:`~kpfm_pl.match.Match` class.
"""

from __future__ import annotations

import warnings
from typing import TypedDict

import numpy as np
from scipy import ndimage

__all__ = [
    "TriggerPoint",
    "feature_scale",
    "smooth",
    "derivative",
    "resample",
    "bin_average",
    "left_bound_fraction",
    "inflection_left_bound",
    "fit_window_mask",
    "lifetime_from_truncated_mean",
    "linear_line_level",
    "set_zero",
    "downsample_sum",
    "nearest_index",
    "trigger_points",
]


class TriggerPoint(TypedDict):
    """One trigger location on a signal axis.

    A single trigger is one such point; a collection of triggers is a plain,
    chronological ``list[TriggerPoint]``.
    """

    index: int
    time: float


def feature_scale(y: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Min-max scale ``y`` to the ``[0, 1]`` range.

    When ``mask`` is given, the min/max are computed over ``y[mask]`` (falling
    back to the full array if the mask is empty) but the map still applies to
    all of ``y``. Robust against constant input (returns zeros instead of NaNs).
    """
    y = np.asarray(y, dtype=float)
    sub = y[mask] if mask is not None and np.any(mask) else y
    lo = float(np.min(sub))
    span = float(np.max(sub) - lo)
    if span == 0 or not np.isfinite(span):
        return np.zeros_like(y)
    return (y - lo) / span


def smooth(y: np.ndarray, window: int = 1, method: str = "moving_average") -> np.ndarray:
    """Smooth a 1-D signal.

    Parameters
    ----------
    y :
        Input signal.
    window :
        Window size (moving average) or sigma (gaussian). ``<= 1`` is a no-op
        for the moving average and ``<= 0`` for the gaussian.
    method :
        ``"moving_average"`` (default) or ``"gaussian"``.
    """
    y = np.asarray(y, dtype=float)
    if method == "moving_average":
        if window <= 1:
            return y
        return ndimage.uniform_filter1d(y, size=int(window), mode="nearest")
    if method == "gaussian":
        if window <= 0:
            return y
        return ndimage.gaussian_filter1d(y, sigma=window)
    raise ValueError(f"Unknown smoothing method: {method!r}")


def derivative(x: np.ndarray, y: np.ndarray, smoothing: int = 1) -> np.ndarray:
    """Numerical derivative ``dy/dx`` using central differences.

    An optional moving-average ``smoothing`` is applied to the result.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dydx = np.gradient(y, x)
    return smooth(dydx, smoothing)


def resample(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    left: float = np.nan,
    right: float = np.nan,
) -> np.ndarray:
    """Linearly interpolate ``(x, y)`` onto a new ``x_new`` grid.

    Values outside the original range default to NaN so they are easy to mask.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    return np.interp(np.asarray(x_new, dtype=float), x[order], y[order], left=left, right=right)


def bin_average(
    x: np.ndarray, y: np.ndarray, bin_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Average ``x`` and ``y`` in consecutive bins of ``bin_size`` samples.

    The final, partially filled bin is averaged over the samples it contains
    (NaN-padded) so trailing data does not bias the mean.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if bin_size <= 1:
        return x, y
    n_bins = int(np.ceil(len(y) / bin_size))
    pad = n_bins * bin_size - len(y)
    xb = np.nanmean(np.pad(x, (0, pad), constant_values=np.nan).reshape(n_bins, bin_size), axis=1)
    yb = np.nanmean(np.pad(y, (0, pad), constant_values=np.nan).reshape(n_bins, bin_size), axis=1)
    return xb, yb


def left_bound_fraction(y: np.ndarray, fraction: float = 0.95, smoothing: float = 2) -> int:
    """Index of the first sample reaching ``fraction`` of the (smoothed) maximum.

    Useful to find the rising edge of a lifetime histogram before fitting.
    """
    y = np.asarray(y, dtype=float)
    if smoothing > 0:
        y = ndimage.gaussian_filter1d(y, sigma=smoothing)
    ymax = np.max(y)
    return int(np.where(y >= fraction * ymax)[0][0])


def inflection_left_bound(y: np.ndarray, smoothing: float = 1.5) -> int | None:
    """Index of the dominant inflection point within the top 10 % of ``y``.

    Returns ``None`` when no curvature sign change is found.
    """
    y = np.asarray(y, dtype=float)
    threshold = 0.10 * np.max(y)
    indices = np.where(y >= threshold)[0]
    if len(indices) < 3:
        return None

    y_smooth = ndimage.gaussian_filter1d(y, sigma=smoothing)
    d2y = np.gradient(np.gradient(y_smooth))
    d2y_sub = d2y[indices]

    crossings = [indices[i] for i in range(1, len(d2y_sub))
                 if np.sign(d2y_sub[i - 1]) != np.sign(d2y_sub[i])]
    if not crossings:
        return None
    if len(crossings) == 1:
        return int(crossings[0])

    # Pick the crossing with the largest curvature jump.
    best, best_jump = crossings[0], -np.inf
    for idx in crossings:
        j = int(np.where(indices == idx)[0][0])
        jump = abs(d2y_sub[j] - d2y_sub[j - 1])
        if jump > best_jump:
            best_jump, best = jump, idx
    return int(best)


def fit_window_mask(
    t_shifted: np.ndarray,
    y: np.ndarray,
    mode: str = "fraction",
    *,
    fraction: float = 0.95,
    smoothing: float = 2.0,
    start: float = 0.0,
    end: float | None = None,
) -> np.ndarray:
    """Boolean mask selecting the fit window of a (peak-shifted) decay curve.

    Shared by the lifetime fit and the FLID map so both pick their fit window
    identically. ``t_shifted`` is the time axis shifted so the peak sits at zero;
    ``y`` is the curve used to locate the left bound.

    ``mode``:

    * ``"fraction"`` -- start at the rising edge reaching ``fraction`` of the
      (``smoothing``-blurred) maximum (see :func:`left_bound_fraction`).
    * ``"inflection"`` -- start at the dominant inflection point
      (see :func:`inflection_left_bound`; falls back to index 0).
    * ``"manual"`` -- keep ``start <= t_shifted <= end`` (``end=None`` uses the
      last sample).
    """
    t_shifted = np.asarray(t_shifted, dtype=float)
    y = np.asarray(y, dtype=float)
    if mode == "fraction":
        lb = left_bound_fraction(y, fraction, smoothing)
        return t_shifted >= t_shifted[lb]
    if mode == "inflection":
        lb = inflection_left_bound(y, smoothing) or 0
        return t_shifted >= t_shifted[lb]
    if mode == "manual":
        end = t_shifted[-1] if end is None else end
        return (t_shifted >= start) & (t_shifted <= end)
    raise ValueError(f"Unknown mode: {mode!r}")


def lifetime_from_truncated_mean(mean: np.ndarray | float, window: np.ndarray | float,
                                 ) -> np.ndarray | float:
    """Lifetime of a single exponential from its mean delay inside a finite window.

    Photons of a decay ``exp(-t/tau)`` observed only for ``0 <= t < T`` have
    the mean delay ``tau - T / (exp(T/tau) - 1)``: the late photons are cut
    off, so the mean is shorter than ``tau`` (by < 0.1 % for ``tau < T/7``,
    about 3 % at ``tau = T/5`` and 22 % at ``tau = T/2.5``). It grows
    monotonically from 0 to ``T/2``, so it is inverted here by bisection.

    Parameters
    ----------
    mean :
        Mean delay(s) after the window start (same unit as ``window``).
    window :
        Window length(s) ``T``.

    Returns
    -------
    float or ndarray
        ``tau``; ``NaN`` where ``mean`` is not inside ``(0, T/2)`` (e.g. a
        background-dominated bin) or not finite.
    """
    scalar = np.ndim(mean) == 0 and np.ndim(window) == 0
    mean, window = np.broadcast_arrays(np.asarray(mean, dtype=float),
                                       np.asarray(window, dtype=float))
    out = np.full(mean.shape, np.nan)
    ok = (np.isfinite(mean) & np.isfinite(window) & (window > 0)
          & (mean > 0) & (mean < window / 2))
    m, t = mean[ok], window[ok]

    def truncated_mean(tau: np.ndarray) -> np.ndarray:
        x = t / tau
        with np.errstate(over="ignore"):
            return tau * (1.0 - x / np.expm1(x))

    # tau >= mean always; for large tau the mean approaches T/2 - T^2/(12 tau).
    lo = np.log(m)
    hi = np.log(np.maximum(4 * t ** 2 / (12 * (t / 2 - m)), 4 * m))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        below = truncated_mean(np.exp(mid)) < m
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    out[ok] = np.exp(0.5 * (lo + hi))
    return float(out) if scalar else out


def linear_line_level(data: np.ndarray, axis: int = 1) -> np.ndarray:
    """Remove a best-fit line from each row (``axis=1``) or column (``axis=0``).

    A common first step when flattening AFM height images.
    """
    data = np.asarray(data, dtype=float)
    leveled = np.empty_like(data)
    if axis == 1:
        x = np.arange(data.shape[1])
        for i in range(data.shape[0]):
            leveled[i, :] = data[i, :] - np.polyval(np.polyfit(x, data[i, :], 1), x)
    elif axis == 0:
        x = np.arange(data.shape[0])
        for j in range(data.shape[1]):
            leveled[:, j] = data[:, j] - np.polyval(np.polyfit(x, data[:, j], 1), x)
    else:
        raise ValueError("axis must be 0 (columns) or 1 (rows)")
    return leveled


def set_zero(data: np.ndarray) -> np.ndarray:
    """Shift ``data`` so its minimum sits at zero."""
    data = np.asarray(data, dtype=float)
    return np.abs(data - data.min())


def downsample_sum(image: np.ndarray, fx: int, fy: int) -> np.ndarray:
    """Downsample ``image`` by summing ``fx`` x ``fy`` pixel blocks.

    Trailing rows/columns that do not fill a full block are dropped. Useful
    to speed up per-pixel fitting on large images.
    """
    image = np.asarray(image)
    h, w = image.shape
    h2 = (h // fx) * fx
    w2 = (w // fy) * fy
    cropped = image[:h2, :w2]
    return cropped.reshape(h2 // fx, fx, w2 // fy, fy).sum(axis=(1, 3))


def nearest_index(axis: np.ndarray, value: float) -> int:
    """Index of the sample in ``axis`` closest to ``value``."""
    axis = np.asarray(axis, dtype=float)
    return int(np.argmin(np.abs(axis - value)))


def trigger_points(times: np.ndarray, axis: np.ndarray) -> list[TriggerPoint]:
    """Map trigger ``times`` onto the samples of a time ``axis``.

    Each time is snapped to its nearest sample, so every returned point
    satisfies ``axis[point["index"]] == point["time"]``. This is the trigger
    format returned by the loaders and consumed by :class:`~kpfm_pl.match.Match`.

    Times more than half a sample step outside ``axis`` are dropped with a
    warning rather than clamped to an unrelated endpoint. Consecutive times
    that snap to the same sample collapse into one point.

    Returns
    -------
    list[TriggerPoint]
        ``[{"index": int, "time": float}, ...]`` in the order of ``times``.
    """
    axis = np.asarray(axis, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
        raise ValueError("axis must be a non-empty finite 1-D time axis.")
    half_step = 0.5 * float(np.median(np.abs(np.diff(axis)))) if axis.size > 1 else 0.0
    inside = (times >= axis.min() - half_step) & (times <= axis.max() + half_step)
    if not np.all(inside):
        warnings.warn(
            f"{int(np.count_nonzero(~inside))} trigger(s) outside the time axis "
            "were dropped.",
            RuntimeWarning,
            stacklevel=3,
        )
    points: list[TriggerPoint] = []
    for t in times[inside]:
        index = nearest_index(axis, t)
        if points and points[-1]["index"] == index:
            continue
        points.append({"index": index, "time": float(axis[index])})
    return points
