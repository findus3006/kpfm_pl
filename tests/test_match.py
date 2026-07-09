import numpy as np
import pytest

from kpfm_pl.core import Signal
from kpfm_pl.match import Match, _robust_trigger_value, _trigger_times


def _triggers(left_idx, left_t, right_idx, right_t):
    return {
        "left": {"index": left_idx, "time": left_t},
        "right": {"index": right_idx, "time": right_t},
    }


def _make_pair(k_y=None, t_y=None):
    x = np.linspace(0.0, 10.0, 101)
    if k_y is None:
        k_y = np.linspace(0.0, 1.0, 101)
    if t_y is None:
        t_y = np.linspace(0.0, 10.0, 101)
    kpfm = Signal(x, k_y, label="kpfm", units={"x": "s", "y": "V"})
    timetrace = Signal(x, t_y, label="tt", units={"x": "s", "y": "counts"})
    return kpfm, timetrace


# --------------------------------------------------------------- helpers
def test_trigger_times_from_dict():
    trig = _triggers(1, 2.0, 3, 8.0)
    assert _trigger_times(trig) == (2.0, 8.0)


def test_trigger_times_from_float():
    assert _trigger_times(3.5) == (3.5, 3.5)


def test_robust_trigger_value_single_sample():
    y = np.array([0.0, 5.0, 10.0])
    assert _robust_trigger_value(y, 1, window=1) == 5.0


def test_robust_trigger_value_median_window():
    y = np.array([0.0, 100.0, 2.0, 3.0, 4.0])
    # median over indices 0..4 suppresses the spike at index 1.
    assert _robust_trigger_value(y, 2, window=2) == 3.0


# --------------------------------------------------------------- scaling
def test_none_scaling_is_identity():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_triggers(20, 2.0, 80, 8.0),
              timetrace_triggers=_triggers(20, 2.0, 80, 8.0),
              scaling="none")
    assert m.kpfm_scale == 1.0 and m.kpfm_shift == 0.0
    assert m.scale == 1.0 and m.shift == 0.0
    native = m.timetrace_signal(on_grid=False)
    assert np.allclose(native["y"], tt["y"])


def test_time_shift_aligns_left_triggers():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_triggers(20, 2.0, 80, 8.0),
              timetrace_triggers=_triggers(25, 2.5, 85, 8.5),
              scaling="none")
    # common frame uses the KPFM trigger times
    assert m.triggers == {"left": 2.0, "right": 8.0}
    # timetrace axis is shifted by (t_left - k_left) = 0.5
    assert np.allclose(m.timetrace_signal(on_grid=False)["x"], tt["x"] - 0.5)


def test_feature_scaling_maps_window_to_unit_range():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_triggers(20, 2.0, 80, 8.0),
              timetrace_triggers=_triggers(20, 2.0, 80, 8.0),
              scaling="feature")
    ks = m.kpfm_signal(on_grid=True)
    win = (ks["x"] >= 2.0) & (ks["x"] <= 8.0)
    y = ks["y"][win]
    assert np.nanmin(y) == pytest.approx(0.0, abs=1e-6)
    assert np.nanmax(y) == pytest.approx(1.0, abs=1e-6)


def test_zscore_scaling_standardizes_window():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_triggers(20, 2.0, 80, 8.0),
              timetrace_triggers=_triggers(20, 2.0, 80, 8.0),
              scaling="zscore")
    ks = m.kpfm_signal(on_grid=False)
    win = (ks["x"] >= 2.0) & (ks["x"] <= 8.0)
    y = ks["y"][win]
    assert np.mean(y) == pytest.approx(0.0, abs=1e-6)
    assert np.std(y) == pytest.approx(1.0, abs=1e-6)


def test_trigger_scaling_matches_anchors():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_triggers(20, 2.0, 80, 8.0),
              timetrace_triggers=_triggers(25, 2.5, 85, 8.5),
              scaling="trigger", trigger_window=1)
    assert m.kpfm_scale == 1.0 and m.kpfm_shift == 0.0
    assert m.scale == pytest.approx(0.1)
    assert m.shift == pytest.approx(-0.05)
    tt_native = m.timetrace_signal(on_grid=False)["y"]
    # scaled time-trace anchors coincide with the KPFM trigger amplitudes
    assert tt_native[25] == pytest.approx(kpfm["y"][20])
    assert tt_native[85] == pytest.approx(kpfm["y"][80])


def test_trigger_scaling_requires_dict_triggers():
    kpfm, tt = _make_pair()
    with pytest.raises(ValueError):
        Match(kpfm, tt, kpfm_triggers=2.0, timetrace_triggers=2.0,
              scaling="trigger")


def test_unknown_scaling_raises():
    kpfm, tt = _make_pair()
    with pytest.raises(ValueError):
        Match(kpfm, tt,
              kpfm_triggers=_triggers(20, 2.0, 80, 8.0),
              timetrace_triggers=_triggers(20, 2.0, 80, 8.0),
              scaling="bogus")


# --------------------------------------------------------------- stats
def test_pearson_perfectly_correlated():
    x = np.linspace(0.0, 10.0, 201)
    y = np.sin(x)
    kpfm, tt = _make_pair(k_y=y, t_y=y)
    kpfm = Signal(x, y)
    tt = Signal(x, y)
    m = Match(kpfm, tt,
              kpfm_triggers=_triggers(40, 2.0, 160, 8.0),
              timetrace_triggers=_triggers(40, 2.0, 160, 8.0),
              scaling="none")
    r = m.pearson()
    assert r["statistic"] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= r["pvalue"] <= 1.0


def test_cross_correlation_zero_lag_for_identical_signals():
    x = np.linspace(0.0, 10.0, 201)
    y = np.sin(x)
    m = Match(Signal(x, y), Signal(x, y),
              kpfm_triggers=_triggers(40, 2.0, 160, 8.0),
              timetrace_triggers=_triggers(40, 2.0, 160, 8.0),
              scaling="none")
    out = m.cross_correlation()
    assert set(out) == {"lag", "correlation", "best_lag"}
    assert out["best_lag"] == pytest.approx(0.0, abs=1e-2)
    assert np.max(out["correlation"]) == pytest.approx(1.0, abs=1e-6)


def test_pearson_full_range_ignores_window():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_triggers(20, 2.0, 80, 8.0),
              timetrace_triggers=_triggers(20, 2.0, 80, 8.0),
              scaling="none")
    full = m.pearson(within_triggers=False)
    windowed = m.pearson(within_triggers=True)
    # both are strong positive correlations of two ramps
    assert full["statistic"] == pytest.approx(1.0, abs=1e-6)
    assert windowed["statistic"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------- output
def test_to_dict_structure():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_triggers(20, 2.0, 80, 8.0),
              timetrace_triggers=_triggers(20, 2.0, 80, 8.0),
              scaling="feature")
    d = m.to_dict()
    assert set(d) >= {"grid", "kpfm", "timetrace", "scaling", "scale",
                      "shift", "kpfm_scale", "kpfm_shift", "triggers"}
    assert d["scaling"] == "feature"
    assert len(d["grid"]) == len(m.grid)


def test_kpfm_derivative_of_ramp_is_constant():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_triggers(20, 2.0, 80, 8.0),
              timetrace_triggers=_triggers(20, 2.0, 80, 8.0),
              scaling="none")
    d = m.kpfm_derivative(on_grid=True)
    # d/dt of a 0..1 ramp over 0..10 s is 0.1 /s
    assert np.allclose(d["y"], 0.1, atol=1e-6)


def test_single_float_trigger_window_falls_back_to_full_overlap():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt, kpfm_triggers=2.0, timetrace_triggers=2.0,
              scaling="feature")
    # left == right -> no laser-on window, mask is the whole grid
    mask = m._window_mask(m.grid)
    assert mask.all()


def test_explicit_n_points_controls_grid_length():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_triggers(20, 2.0, 80, 8.0),
              timetrace_triggers=_triggers(20, 2.0, 80, 8.0),
              n_points=50, scaling="none")
    assert len(m.grid) == 50
