"""Reusable, dependency-light signal-processing utilities.

These functions are pure (no I/O, no plotting) so they are easy to test and to
reuse across the loaders and the :class:`~kpfm_pl.match.Match` class.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "feature_scale",
    "feature_scale_params",
    "zscore",
    "zscore_params",
    "smooth",
    "derivative",
    "resample",
    "bin_average",
    "left_bound_fraction",
    "inflection_left_bound",
    "linear_line_level",
    "set_zero",
    "downsample_sum",
]


def feature_scale_params(
    y: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float]:
    """``(scale, shift)`` of the affine map that min-max scales ``y`` to ``[0, 1]``.

    The returned coefficients satisfy ``y_scaled = scale * y + shift``. When
    ``mask`` is given, the min/max are computed over ``y[mask]`` (falling back to
    the full array if the mask is empty) but the map still applies to all of
    ``y``. Constant input yields ``(0.0, 0.0)`` (i.e. all zeros, no NaNs).
    """
    y = np.asarray(y, dtype=float)
    sub = y[mask] if mask is not None and np.any(mask) else y
    lo = float(np.min(sub))
    span = float(np.max(sub) - lo)
    if span == 0 or not np.isfinite(span):
        return 0.0, 0.0
    return 1.0 / span, -lo / span


def feature_scale(y: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Min-max scale ``y`` to the ``[0, 1]`` range.

    Robust against constant input (returns zeros instead of NaNs). See
    :func:`feature_scale_params` for the optional ``mask`` semantics.
    """
    scale, shift = feature_scale_params(y, mask)
    return scale * np.asarray(y, dtype=float) + shift


def zscore_params(
    y: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float]:
    """``(scale, shift)`` of the affine map that standardizes ``y``.

    The returned coefficients satisfy ``y_scaled = scale * y + shift`` and map
    ``y`` to zero mean / unit standard deviation. When ``mask`` is given, the
    mean/std are computed over ``y[mask]`` (falling back to the full array if the
    mask is empty). Constant input yields ``(0.0, 0.0)``.
    """
    y = np.asarray(y, dtype=float)
    sub = y[mask] if mask is not None and np.any(mask) else y
    mean = float(np.mean(sub))
    std = float(np.std(sub))
    if std == 0 or not np.isfinite(std):
        return 0.0, 0.0
    return 1.0 / std, -mean / std


def zscore(y: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Standardize ``y`` to zero mean / unit standard deviation.

    Robust against constant input (returns zeros). See :func:`zscore_params` for
    the optional ``mask`` semantics.
    """
    scale, shift = zscore_params(y, mask)
    return scale * np.asarray(y, dtype=float) + shift


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
