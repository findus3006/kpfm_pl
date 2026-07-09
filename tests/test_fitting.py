import numpy as np
import pytest

from kpfm_pl.fitting import MultiExpFit


def _single_exp(t, A, tau, C=0.0):
    return A * np.exp(-t / tau) + C


def test_single_exponential_recovers_parameters():
    t = np.linspace(0.0, 50.0, 500)
    y = _single_exp(t, A=1.0, tau=8.0)
    fit = MultiExpFit(t, y, n_exp=1, offset=0.0)
    params = fit.parameters()
    assert fit.success
    assert params["tau"][0] == pytest.approx(8.0, rel=1e-3)
    assert params["A"][0] == pytest.approx(1.0, rel=1e-3)


def test_offset_is_respected():
    t = np.linspace(0.0, 50.0, 500)
    y = _single_exp(t, A=1.0, tau=5.0, C=0.2)
    fit = MultiExpFit(t, y, n_exp=1, offset=0.2)
    assert fit.parameters()["tau"][0] == pytest.approx(5.0, rel=1e-2)


def test_double_exponential():
    t = np.linspace(0.0, 60.0, 800)
    y = _single_exp(t, 0.7, 3.0) + _single_exp(t, 0.3, 20.0)
    fit = MultiExpFit(t, y, n_exp=2, offset=0.0)
    taus = np.sort(fit.parameters()["tau"])
    assert taus[0] == pytest.approx(3.0, rel=5e-2)
    assert taus[1] == pytest.approx(20.0, rel=5e-2)


def test_r_squared_high_for_clean_data():
    t = np.linspace(0.0, 40.0, 400)
    y = _single_exp(t, 1.0, 6.0)
    fit = MultiExpFit(t, y, n_exp=1)
    assert fit.quality()["r_squared"] > 0.999


def test_curve_matches_data_shape():
    t = np.linspace(0.0, 40.0, 123)
    y = _single_exp(t, 1.0, 6.0)
    fit = MultiExpFit(t, y, n_exp=1)
    assert fit.curve().shape == t.shape


def test_constrain_sum_a():
    t = np.linspace(0.0, 40.0, 400)
    y = _single_exp(t, 0.6, 4.0) + _single_exp(t, 0.3, 15.0)
    fit = MultiExpFit(t, y, n_exp=2, constrain_sum_A=True)
    assert np.sum(fit.parameters()["A"]) <= 1.0 + 1e-6


def test_to_dict_keys():
    t = np.linspace(0.0, 40.0, 200)
    y = _single_exp(t, 1.0, 5.0)
    out = MultiExpFit(t, y, n_exp=1).to_dict()
    assert {"A", "tau", "offset", "curve", "quality", "label", "success"} <= set(out)


def test_legend_label_is_latex():
    t = np.linspace(0.0, 40.0, 200)
    y = _single_exp(t, 1.0, 5.0)
    label = MultiExpFit(t, y, n_exp=1).legend_label()
    assert label.startswith("$f(t) = ") and label.endswith("$")


def test_failed_fit_warns_and_raises_clear_error():
    t = np.linspace(0.0, 10.0, 50)
    y = _single_exp(t, 1.0, 3.0)
    # ``A = -1`` violates the lower bound (0), so ``least_squares`` rejects the
    # initial guess as infeasible and the fit fails.
    with pytest.warns(RuntimeWarning, match="fit failed"):
        fit = MultiExpFit(t, y, n_exp=1, p0=np.array([-1.0, 3.0]))

    assert not fit.success
    # Accessors must surface a clear RuntimeError (carrying the cause), not an
    # opaque AttributeError from dereferencing a missing result.
    with pytest.raises(RuntimeError):
        fit.parameters()
    with pytest.raises(RuntimeError):
        fit.curve()
    with pytest.raises(RuntimeError):
        fit.quality()
    with pytest.raises(RuntimeError):
        fit.to_dict()
