import numpy as np
import pytest

from kpfm_pl.utils import (
    bin_average,
    derivative,
    downsample_sum,
    feature_scale,
    fit_window_mask,
    inflection_left_bound,
    left_bound_fraction,
    linear_line_level,
    nearest_index,
    resample,
    set_zero,
    smooth,
    trigger_points,
)


def test_feature_scale_range():
    y = np.array([2.0, 4.0, 6.0])
    scaled = feature_scale(y)
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0


def test_feature_scale_constant_input():
    assert np.all(feature_scale(np.full(5, 3.0)) == 0.0)


def test_smooth_moving_average_noop():
    y = np.arange(10.0)
    assert np.array_equal(smooth(y, window=1), y)


def test_smooth_reduces_variance():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    assert smooth(y, window=20).var() < y.var()


def test_derivative_of_line_is_slope():
    x = np.linspace(0, 10, 101)
    y = 3.0 * x + 1.0
    d = derivative(x, y)
    assert np.allclose(d, 3.0, atol=1e-6)


def test_resample_linear():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0])
    out = resample(x, y, np.array([0.5, 1.5]))
    assert np.allclose(out, [5.0, 15.0])


def test_resample_out_of_range_is_nan():
    out = resample(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([-1.0, 2.0]))
    assert np.all(np.isnan(out))


def test_bin_average_even():
    x = np.arange(6.0)
    y = np.arange(6.0)
    xb, yb = bin_average(x, y, 2)
    assert np.allclose(yb, [0.5, 2.5, 4.5])


def test_bin_average_trailing_partial_bin():
    y = np.array([1.0, 1.0, 1.0])
    _, yb = bin_average(np.arange(3.0), y, 2)
    assert np.allclose(yb, [1.0, 1.0])


def test_left_bound_fraction_finds_rising_edge():
    y = np.concatenate([np.zeros(10), np.linspace(0, 1, 10), np.ones(10)])
    idx = left_bound_fraction(y, fraction=0.95, smoothing=0)
    assert idx >= 18


def test_linear_line_level_removes_tilt():
    ramp = np.tile(np.linspace(0, 1, 20), (5, 1))
    leveled = linear_line_level(ramp, axis=1)
    assert np.allclose(leveled, 0.0, atol=1e-9)


def test_set_zero_minimum():
    assert set_zero(np.array([-3.0, -1.0, 2.0])).min() == 0.0


def _decay_curve():
    # Peak at index 20, then exponential decay; time axis shifted so peak is at 0.
    t = np.linspace(0, 40, 200)
    peak = 20
    y = np.zeros_like(t)
    y[peak:] = np.exp(-(t[peak:] - t[peak]) / 5.0)
    y[:peak] = np.linspace(0, 1, peak)
    t_shifted = t - t[peak]
    return t_shifted, y


def test_fit_window_mask_fraction_matches_left_bound_fraction():
    t_shifted, y = _decay_curve()
    lb = left_bound_fraction(y, fraction=0.95, smoothing=2.0)
    mask = fit_window_mask(t_shifted, y, "fraction", fraction=0.95, smoothing=2.0)
    assert np.array_equal(mask, t_shifted >= t_shifted[lb])


def test_fit_window_mask_manual_bounds():
    t_shifted, y = _decay_curve()
    mask = fit_window_mask(t_shifted, y, "manual", start=0.0, end=5.0)
    assert np.array_equal(mask, (t_shifted >= 0.0) & (t_shifted <= 5.0))


def test_fit_window_mask_manual_default_end_is_last_sample():
    t_shifted, y = _decay_curve()
    mask = fit_window_mask(t_shifted, y, "manual", start=0.0)
    assert np.array_equal(mask, t_shifted >= 0.0)


def test_fit_window_mask_unknown_mode_raises():
    t_shifted, y = _decay_curve()
    with pytest.raises(ValueError):
        fit_window_mask(t_shifted, y, "nonsense")


def test_nearest_index_exact_and_between():
    axis = np.array([0.0, 1.0, 2.0, 3.0])
    assert nearest_index(axis, 2.0) == 2
    assert nearest_index(axis, 2.4) == 2
    assert nearest_index(axis, 2.6) == 3


def test_trigger_points_snap_to_axis_samples():
    axis = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    points = trigger_points(np.array([0.2, 0.8]), axis)
    assert points == [{"index": 1, "time": 0.25}, {"index": 3, "time": 0.75}]
    assert isinstance(points[0]["index"], int)
    assert isinstance(points[0]["time"], float)


def test_trigger_points_drop_times_outside_axis():
    axis = np.array([0.5, 0.75])
    # 0.8 is within half a step of the axis end and is kept; 0.2 is not
    with pytest.warns(RuntimeWarning, match="1 trigger"):
        points = trigger_points(np.array([0.2, 0.8]), axis)
    assert points == [{"index": 1, "time": 0.75}]


def test_trigger_points_collapse_consecutive_duplicates():
    axis = np.arange(5.0)
    points = trigger_points(np.array([1.0, 1.1, 3.0]), axis)
    assert [p["index"] for p in points] == [1, 3]


def test_trigger_points_single_time():
    assert trigger_points(2.0, np.arange(5.0)) == [{"index": 2, "time": 2.0}]


def test_feature_scale_with_mask():
    y = np.array([0.0, 10.0, 20.0, 30.0])
    mask = np.array([False, True, True, False])
    # min/max taken over the masked window (10..20), map applied to all of y
    assert np.allclose(feature_scale(y, mask), [-1.0, 0.0, 1.0, 2.0])


def test_feature_scale_empty_mask_falls_back_to_full():
    y = np.array([2.0, 4.0, 6.0])
    mask = np.zeros(3, bool)
    assert np.allclose(feature_scale(y, mask), [0.0, 0.5, 1.0])


def test_smooth_gaussian_reduces_variance():
    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    assert smooth(y, window=5, method="gaussian").var() < y.var()


def test_smooth_gaussian_noop_for_nonpositive_window():
    y = np.arange(5.0)
    assert np.array_equal(smooth(y, window=0, method="gaussian"), y)


def test_smooth_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown smoothing method"):
        smooth(np.arange(5.0), window=2, method="bogus")


def test_derivative_with_smoothing_runs():
    x = np.linspace(0, 10, 101)
    y = 2.0 * x
    d = derivative(x, y, smoothing=3)
    assert np.allclose(d, 2.0, atol=1e-6)


def test_bin_average_noop_for_small_bin():
    x = np.arange(4.0)
    y = np.arange(4.0)
    xb, yb = bin_average(x, y, 1)
    assert np.array_equal(xb, x) and np.array_equal(yb, y)


def test_left_bound_fraction_with_smoothing():
    y = np.concatenate([np.zeros(10), np.linspace(0, 1, 10), np.ones(10)])
    idx = left_bound_fraction(y, fraction=0.9, smoothing=2)
    assert 10 <= idx <= 29


def test_inflection_left_bound_detects_curvature_change():
    x = np.linspace(-5, 5, 200)
    y = 1.0 / (1.0 + np.exp(-x))  # sigmoid: inflection near the centre
    idx = inflection_left_bound(y, smoothing=1.5)
    assert idx is not None
    assert 0 <= idx < len(y)


def test_inflection_left_bound_returns_none_for_short_peak():
    y = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    assert inflection_left_bound(y) is None


def test_linear_line_level_axis0_removes_tilt():
    ramp = np.tile(np.linspace(0, 1, 5).reshape(-1, 1), (1, 4))
    leveled = linear_line_level(ramp, axis=0)
    assert np.allclose(leveled, 0.0, atol=1e-9)


def test_linear_line_level_invalid_axis_raises():
    with pytest.raises(ValueError, match="axis must be"):
        linear_line_level(np.ones((3, 3)), axis=2)


def test_downsample_sum_blocks():
    image = np.ones((4, 6))
    out = downsample_sum(image, 2, 3)
    assert out.shape == (2, 2)
    assert np.all(out == 6.0)


def test_downsample_sum_drops_trailing_partial_blocks():
    image = np.ones((5, 5))
    out = downsample_sum(image, 2, 2)
    assert out.shape == (2, 2)
    assert np.all(out == 4.0)


def test_lifetime_from_truncated_mean_inverts_the_truncated_mean():
    from kpfm_pl.utils import lifetime_from_truncated_mean

    window = 50.0
    taus = np.array([0.5, 2.0, 10.0, 20.0, 200.0])
    means = taus - window / np.expm1(window / taus)
    np.testing.assert_allclose(lifetime_from_truncated_mean(means, window), taus, rtol=1e-9)
    # A mean at or above T/2 (flat background) or <= 0 has no exponential lifetime.
    assert np.isnan(lifetime_from_truncated_mean(np.array([0.0, 25.0, 30.0, np.nan]), window)).all()
    assert isinstance(lifetime_from_truncated_mean(3.0, window), float)
