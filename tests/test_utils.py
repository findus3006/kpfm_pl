import numpy as np
import pytest

from kpfm_pl.utils import (
    bin_average,
    derivative,
    feature_scale,
    left_bound_fraction,
    linear_line_level,
    resample,
    set_zero,
    smooth,
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
