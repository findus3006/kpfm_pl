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


def test_r_squared_uses_unweighted_residuals_for_weighted_fits():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 40.0, 400)
    y = _single_exp(t, 1.0, 6.0) + rng.normal(0.0, 0.02, t.size)
    weights = np.linspace(1.0, 5.0, t.size)
    fit = MultiExpFit(t, y, n_exp=1, weights=weights)
    q = fit.quality()
    expected = 1 - np.sum((fit.curve() - y) ** 2) / np.sum((y - y.mean()) ** 2)
    assert q["r_squared"] == pytest.approx(expected)
    assert 0.0 <= q["r_squared"] <= 1.0
    # "residuals" stays the optimiser's weighted residual vector
    assert len(q["residuals"]) == t.size


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


def test_nan_guesses_are_filled_and_clipped_into_bounds():
    t = np.linspace(0.0, 50.0, 400)
    y = _single_exp(t, A=2.0, tau=8.0)
    # tau guess given, amplitude automatic; bounds exclude the automatic tau guess
    fit = MultiExpFit(t, y, n_exp=1, p0=np.array([np.nan, 6.0]),
                      bounds=(np.array([0.0, 5.0]), np.array([np.inf, 20.0])))
    assert fit.success
    assert fit.parameters()["tau"][0] == pytest.approx(8.0, rel=1e-3)
    # Fully automatic guesses are clipped into user bounds instead of failing.
    fit = MultiExpFit(t, y, n_exp=1, bounds=(np.array([0.0, 7.5]), np.array([np.inf, 9.0])))
    assert fit.success
    assert fit.parameters()["tau"][0] == pytest.approx(8.0, rel=1e-3)


def test_poisson_chi_squared():
    from kpfm_pl import poisson_chi_squared

    counts = np.array([100.0, 49.0, 0.0])
    model = np.array([110.0, 42.0, 1.0])
    q = poisson_chi_squared(counts, model, n_params=1)
    np.testing.assert_allclose(q["weighted_residuals"], [-1.0, 1.0, -1.0])
    assert q["chi_squared"] == pytest.approx(3.0)
    assert q["dof"] == 2 and q["reduced_chi_squared"] == pytest.approx(1.5)
    assert np.isnan(poisson_chi_squared(counts, model, n_params=3)["reduced_chi_squared"])
    with pytest.raises(ValueError):
        poisson_chi_squared(counts, model[:2], n_params=1)


def test_poisson_noise_gives_reduced_chi_squared_near_one():
    from kpfm_pl import poisson_chi_squared

    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 40.0, 2000)
    model = _single_exp(t, A=500.0, tau=6.0, C=20.0)
    counts = rng.poisson(model).astype(float)
    assert poisson_chi_squared(counts, model, n_params=2)["reduced_chi_squared"] \
        == pytest.approx(1.0, abs=0.1)


def test_covariance_matches_curve_fit_and_gives_tau_int_error():
    from scipy.optimize import curve_fit

    rng = np.random.default_rng(1)
    t = np.linspace(0.0, 40.0, 400)
    y = _single_exp(t, 0.6, 2.0) + _single_exp(t, 0.4, 9.0) + rng.normal(0, 0.005, t.size)
    fit = MultiExpFit(t, y, n_exp=2)
    p = fit.parameters()
    popt, pcov = curve_fit(lambda tt, a1, a2, t1, t2: a1 * np.exp(-tt / t1) + a2 * np.exp(-tt / t2),
                           t, y, p0=np.concatenate([p["A"], p["tau"]]))
    np.testing.assert_allclose(fit.covariance(), pcov, rtol=1e-3, atol=1e-12)

    out = fit.to_dict()
    A, tau = p["A"], p["tau"]
    assert out["tau_int"] == pytest.approx(np.sum(A * tau**2) / np.sum(A * tau))
    assert out["tau_int"] == pytest.approx((0.6 * 4 + 0.4 * 81) / (0.6 * 2 + 0.4 * 9), rel=1e-2)
    assert 0 < out["tau_int_err"] < 0.1
    np.testing.assert_allclose(out["tau_err"], np.sqrt(np.diag(pcov))[2:], rtol=1e-3)


def test_single_exponential_tau_int_equals_tau():
    rng = np.random.default_rng(2)
    t = np.linspace(0.0, 30.0, 300)
    fit = MultiExpFit(t, _single_exp(t, 1.0, 4.0) + rng.normal(0, 0.01, t.size), n_exp=1)
    out = fit.to_dict()
    assert out["tau_int"] == pytest.approx(out["tau"][0])
    assert out["tau_int_err"] == pytest.approx(out["tau_err"][0])


# ------------------------------------------------- BIC, deviance and warnings
def test_poisson_chi_squared_reports_deviance_and_bic():
    from kpfm_pl import poisson_chi_squared, poisson_deviance

    counts = np.array([100.0, 49.0, 0.0])
    model = np.array([110.0, 42.0, 1.0])
    deviance = 2 * ((110 - 100 + 100 * np.log(100 / 110)) + (42 - 49 + 49 * np.log(49 / 42))
                    + 1.0)
    assert poisson_deviance(counts, model) == pytest.approx(deviance)
    q = poisson_chi_squared(counts, model, n_params=1)
    assert q["deviance"] == pytest.approx(deviance)
    assert q["n_points"] == 3 and q["n_params"] == 1
    assert q["bic_statistic"] == "chi_squared"
    assert q["bic"] == pytest.approx(3.0 + np.log(3))
    q = poisson_chi_squared(counts, model, n_params=2, bic_from="deviance")
    assert q["bic"] == pytest.approx(deviance + 2 * np.log(3))
    with pytest.raises(ValueError, match="bic_from"):
        poisson_chi_squared(counts, model, n_params=1, bic_from="aic")


def test_bic_prefers_the_true_number_of_components():
    from kpfm_pl import poisson_chi_squared

    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 60.0, 600)
    truth = _single_exp(t, 3000.0, 2.0) + _single_exp(t, 1000.0, 12.0) + 5.0
    counts = rng.poisson(truth).astype(float)
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    bic = {}
    for n in (1, 2, 3):
        fit = MultiExpFit(t, counts, n, offset=5.0, weights=weights)
        bic[n] = poisson_chi_squared(counts, fit.curve(), 2 * n)["bic"]
    assert bic[2] < bic[1] - 10      # the second component is clearly needed
    assert bic[3] > bic[2] - 10      # a third one is not supported


def test_fit_warnings_detects_collapse_merge_and_bounds():
    from kpfm_pl import fit_warnings

    assert fit_warnings([0.6, 0.4], [2.0, 9.0], [0.01, 0.01], [0.05, 0.1]) == []
    zero = fit_warnings([0.6, 0.01], [2.0, 9.0], [0.01, 0.02], [0.05, 3.0])
    assert len(zero) == 1 and "compatible with zero" in zero[0]
    tiny = fit_warnings([1.0, 1e-3], [5.0, 1.0], [0.01, 1e-5], [0.05, 0.1])
    assert len(tiny) == 1 and "collapsed" in tiny[0]
    merged = fit_warnings([0.5, 0.5], [3.0, 3.1], [0.01, 0.01], [0.5, 0.5])
    assert len(merged) == 1 and "merged" in merged[0]
    # relative criterion without errors
    assert any("merged" in w for w in fit_warnings([0.5, 0.5], [3.0, 3.2], merge_rel=0.1))
    assert not fit_warnings([0.5, 0.5], [3.0, 3.2], merge_rel=0.05)
    bounds = (np.array([0.0, 0.0, 0.1, 0.1]), np.array([np.inf, np.inf, 10.0, np.inf]))
    at_bound = fit_warnings([0.5, 0.5], [10.0, 1.0], [0.01, 0.01], [0.1, 0.1],
                            fitted=np.array([0.5, 0.5, 10.0, 1.0]), bounds=bounds)
    assert len(at_bound) == 1 and "τ1" in at_bound[0] and "upper bound" in at_bound[0]


def test_multiexpfit_diagnostics_flags_an_unneeded_component():
    rng = np.random.default_rng(4)
    t = np.linspace(0.0, 40.0, 400)
    counts = rng.poisson(_single_exp(t, 2000.0, 5.0)).astype(float)
    fit = MultiExpFit(t, counts, 2, weights=1 / np.sqrt(np.maximum(counts, 1)))
    assert fit.diagnostics()
    single = MultiExpFit(t, counts, 1, weights=1 / np.sqrt(np.maximum(counts, 1)))
    assert single.diagnostics() == []


# ------------------------------------------------------------- Poisson MLE
def test_poisson_mle_minimises_the_deviance():
    from kpfm_pl import poisson_deviance

    rng = np.random.default_rng(5)
    t = np.linspace(0.0, 30.0, 300)
    counts = rng.poisson(_single_exp(t, 20.0, 6.0, C=0.5)).astype(float)
    neyman = MultiExpFit(t, counts, 1, offset=0.5,
                         weights=1 / np.sqrt(np.maximum(counts, 1)))
    mle = MultiExpFit(t, counts, 1, offset=0.5, statistic="poisson_mle")
    assert mle.success
    assert poisson_deviance(counts, mle.curve()) <= poisson_deviance(counts, neyman.curve())
    assert mle.parameters()["tau"][0] == pytest.approx(6.0, rel=0.1)
    # the deviance is stationary at the MLE (central differences in A and tau)
    p = mle._require_result().x
    dev0 = poisson_deviance(counts, mle.model(t, p))
    for k in (0, 1):
        dp = np.zeros_like(p)
        dp[k] = 1e-4 * p[k]
        grad = (poisson_deviance(counts, mle.model(t, p + dp))
                - poisson_deviance(counts, mle.model(t, p - dp))) / (2 * dp[k])
        assert abs(grad * p[k]) < 1e-3 * dev0


def test_poisson_mle_with_count_scale_matches_raw_counts():
    rng = np.random.default_rng(6)
    t = np.linspace(0.0, 30.0, 300)
    counts = rng.poisson(_single_exp(t, 50.0, 4.0, C=1.0)).astype(float)
    raw = MultiExpFit(t, counts, 1, offset=1.0, statistic="poisson_mle")
    peak = counts.max()
    norm = MultiExpFit(t, counts / peak, 1, offset=1.0 / peak, statistic="poisson_mle",
                       count_scale=peak)
    assert norm.parameters()["tau"][0] == pytest.approx(raw.parameters()["tau"][0], rel=1e-4)
    assert norm.parameters()["A"][0] * peak == pytest.approx(raw.parameters()["A"][0],
                                                             rel=1e-4)


def test_poisson_mle_argument_checks():
    t = np.linspace(0.0, 10.0, 50)
    y = _single_exp(t, 10.0, 2.0)
    with pytest.raises(ValueError, match="statistic"):
        MultiExpFit(t, y, 1, statistic="mle")
    with pytest.raises(ValueError, match="loss"):
        MultiExpFit(t, y, 1, statistic="poisson_mle", loss="soft_l1")
    with pytest.raises(ValueError, match="weights"):
        MultiExpFit(t, y, 1, statistic="poisson_mle", weights=np.ones(t.size))
    with pytest.raises(ValueError, match="count_scale"):
        MultiExpFit(t, y, 1, statistic="poisson_mle", count_scale=0.0)


# ------------------------------------------------ incomplete-decay correction
def _periodic(t, amps, taus, period):
    """Periodic excitation: each bin also holds the decays of all earlier pulses."""
    return sum(a * np.exp(-t / tau) / (1 - np.exp(-period / tau)) for a, tau in zip(amps, taus))


def test_period_corrects_amplitudes_for_incomplete_decay():
    period = 25.0
    t = np.linspace(0.0, 20.0, 400)
    y = _periodic(t, [0.7, 0.3], [2.0, 15.0], period)
    out = MultiExpFit(t, y, 2, period=period).to_dict()
    order = np.argsort(out["tau"])
    np.testing.assert_allclose(out["tau"][order], [2.0, 15.0], rtol=1e-4)
    np.testing.assert_allclose(out["A"][order], [0.7, 0.3], rtol=1e-4)
    np.testing.assert_allclose(out["A_fit"][order],
                               [0.7 / (1 - np.exp(-25 / 2)), 0.3 / (1 - np.exp(-25 / 15))],
                               rtol=1e-4)
    expected = (0.7 * 4 + 0.3 * 225) / (0.7 * 2 + 0.3 * 15)
    assert out["tau_int"] == pytest.approx(expected, rel=1e-4)
    assert out["tau_int_fit"] > out["tau_int"]      # the long component was inflated
    assert out["period"] == period
    # without a period nothing changes
    plain = MultiExpFit(t, y, 2).to_dict()
    np.testing.assert_allclose(plain["A"], plain["A_fit"])


def test_corrected_covariance_matches_numerical_propagation():
    rng = np.random.default_rng(7)
    period = 30.0
    t = np.linspace(0.0, 25.0, 300)
    y = _periodic(t, [1.0], [12.0], period) + rng.normal(0, 0.01, t.size)
    fit = MultiExpFit(t, y, 1, period=period)
    A_fit, tau = fit._require_result().x
    cov = fit.covariance()
    eps = 1e-6
    grad = np.array([1 - np.exp(-period / tau),
                     A_fit * ((1 - np.exp(-period / (tau + eps)))
                              - (1 - np.exp(-period / (tau - eps)))) / (2 * eps)])
    assert fit.corrected_covariance()[0, 0] == pytest.approx(grad @ cov @ grad, rel=1e-4)
    assert fit.corrected_covariance()[1, 1] == pytest.approx(cov[1, 1])
    with pytest.raises(ValueError, match="period"):
        MultiExpFit(t, y, 1, period=0.0)


# ------------------------------------------------- fractions and mean lifetimes
def test_fraction_and_mean_lifetime_helpers():
    from kpfm_pl import amplitude_fractions, intensity_fractions, mean_lifetime

    A, tau = np.array([0.6, 0.4]), np.array([2.0, 9.0])
    np.testing.assert_allclose(amplitude_fractions(A), [0.6, 0.4])
    np.testing.assert_allclose(intensity_fractions(A, tau), [1.2 / 4.8, 3.6 / 4.8])
    assert mean_lifetime(A, tau) == pytest.approx((0.6 * 4 + 0.4 * 81) / 4.8)
    assert mean_lifetime(A, tau, "amplitude") == pytest.approx(4.8)
    # row-wise on many fits; zero weights and NaN rows give NaN
    rows_A = np.array([[0.6, 0.4], [0.0, 0.0], [np.nan, 1.0]])
    rows_tau = np.array([[2.0, 9.0], [1.0, 2.0], [1.0, 2.0]])
    means = mean_lifetime(rows_A, rows_tau, "amplitude")
    assert means[0] == pytest.approx(4.8) and np.isnan(means[1:]).all()
    assert np.isnan(amplitude_fractions(rows_A)[1]).all()
    with pytest.raises(ValueError, match="kind"):
        mean_lifetime(A, tau, "number")


def test_multiexpfit_fractions_and_tau_amp_with_errors():
    rng = np.random.default_rng(8)
    t = np.linspace(0.0, 40.0, 400)
    y = _single_exp(t, 0.6, 2.0) + _single_exp(t, 0.4, 9.0) + rng.normal(0, 0.005, t.size)
    fit = MultiExpFit(t, y, n_exp=2)
    out = fit.to_dict()
    A, tau = out["A"], out["tau"]
    np.testing.assert_allclose(out["f_amp"], A / A.sum())
    np.testing.assert_allclose(out["f_int"], A * tau / np.sum(A * tau))
    assert out["tau_amp"] == pytest.approx(np.sum(A * tau) / A.sum())
    assert out["f_amp"].sum() == pytest.approx(1.0) and out["f_int"].sum() == pytest.approx(1.0)
    # errors: numerical propagation through the covariance
    p = fit._require_result().x
    cov = fit.covariance()

    def numeric(func, eps=1e-7):
        grad = np.array([(func(p + e) - func(p - e)) / (2 * eps)
                         for e in np.eye(p.size) * eps])
        return np.sqrt(np.einsum("i...,ij,j...->...", grad, cov, grad))

    np.testing.assert_allclose(out["f_amp_err"], numeric(lambda q: q[:2] / q[:2].sum()),
                               rtol=1e-4)
    np.testing.assert_allclose(out["f_int_err"],
                               numeric(lambda q: q[:2] * q[2:] / np.sum(q[:2] * q[2:])),
                               rtol=1e-4)
    assert out["tau_amp_err"] == pytest.approx(
        float(numeric(lambda q: np.sum(q[:2] * q[2:]) / q[:2].sum())), rel=1e-4)
    assert fit.mean_lifetime("intensity") == fit.intensity_weighted_lifetime()
    with pytest.raises(ValueError, match="kind"):
        fit.fractions("number")


# ------------------------------------------------------------ IRF reconvolution
def test_exp_kernel_matches_brute_force_integration():
    from kpfm_pl.fitting import exp_kernel

    dt, tau, n, sub = 0.1, 0.25, 30, 400
    # IRF: one bin of unit area; decay exp(-t/tau) with amplitude 1 count per bin
    src = (np.arange(sub) + 0.5) * dt / sub
    t = (np.arange(n * sub) + 0.5) * dt / sub
    lag = t[:, None] - src[None, :]
    density = np.where(lag >= 0, np.exp(-np.clip(lag, 0, None) / tau), 0.0).mean(axis=1) / dt
    brute = density.reshape(n, sub).sum(axis=1) * dt / sub
    kernel = exp_kernel(tau, dt, n, periodic=False)
    np.testing.assert_allclose(kernel, brute, atol=2e-3 * brute.max())
    assert kernel.sum() == pytest.approx(tau / dt, rel=1e-4)   # tail beyond n bins
    # periodic: the tails of all earlier pulses add up, the total stays tau/dt
    assert exp_kernel(5.0, dt, n, periodic=True).sum() == pytest.approx(50.0)
    assert exp_kernel(5.0, dt, n, periodic=True)[5] > exp_kernel(5.0, dt, n, periodic=False)[5]


def test_prepare_irf_and_fwhm():
    from kpfm_pl.fitting import irf_fwhm, prepare_irf

    t = np.arange(200) * 0.05
    irf = 1000 * np.exp(-0.5 * ((t - 3) / 0.2) ** 2) + 10
    g, background = prepare_irf(irf)
    # the floor bins (below median + 5 sqrt(median)) include a few Gaussian wings
    assert background == pytest.approx(10.0, abs=0.3) and g.sum() == pytest.approx(1.0)
    assert irf_fwhm(g) * 0.05 == pytest.approx(2.3548 * 0.2, rel=0.02)
    assert prepare_irf(irf, None)[1] == 0.0 and prepare_irf(irf, 7.5)[1] == 7.5
    with pytest.raises(ValueError, match="no counts"):
        prepare_irf(np.full(20, 5.0))


def test_prepare_irf_keeps_the_zero_mean_floor_noise():
    # Clipping the floor at zero would keep only its positive noise (v1.7.2 fix).
    from kpfm_pl.fitting import prepare_irf

    rng = np.random.default_rng(10)
    t = np.arange(2000) * 0.05
    irf = rng.poisson(5000 * np.exp(-0.5 * ((t - 10) / 0.2) ** 2) + 3.0).astype(float)
    g, background = prepare_irf(irf)
    assert background == pytest.approx(3.0, abs=0.1)        # the median would give 3
    floor = g[np.abs(t - 10) > 3] * irf.sum()
    assert np.min(floor) < 0 and abs(np.mean(floor)) < 0.1   # noise kept, mean ~0


def test_noisy_irf_floor_does_not_bias_the_fitted_background():
    from kpfm_pl import IRFReconvolutionFit

    rng = np.random.default_rng(11)
    t, clean = _irf_setup()                                  # floor of 20 counts per bin
    noisy = rng.poisson(clean).astype(float)
    truth = IRFReconvolutionFit(clean, np.ones(t.size), 1, 0.1, shift=0.1, background=3.0)
    counts = truth.full_model(np.array([3000.0, 4.0]))       # noise-free decay
    fit = IRFReconvolutionFit(noisy, counts, 1, 0.1, fit_background=True)
    # Clipping the IRF floor (before v1.7.2) gave B = 0.33 here instead of 3.
    assert fit.to_dict()["background"] == pytest.approx(3.0, abs=0.15)
    assert fit.parameters()["tau"][0] == pytest.approx(4.0, rel=0.01)


def _irf_setup(n=1000, dt=0.1, centre=5.0, sigma=0.15):
    t = np.arange(n) * dt
    return t, np.exp(-0.5 * ((t - centre) / sigma) ** 2) * 2e4 + 20


def test_reconvolution_recovers_parameters_from_noise_free_data():
    from kpfm_pl import IRFReconvolutionFit

    t, irf = _irf_setup()
    truth = IRFReconvolutionFit(irf, np.ones(t.size), 2, 0.1, shift=0.23, background=3.0,
                                scatter_counts=2000.0)
    counts = truth.full_model(np.array([3000.0, 800.0, 0.6, 4.0]))
    fit = IRFReconvolutionFit(irf, counts, 2, 0.1, background=3.0, scatter=True)
    out = fit.to_dict()
    order = np.argsort(out["tau"])
    np.testing.assert_allclose(out["tau"][order], [0.6, 4.0], rtol=1e-4)
    np.testing.assert_allclose(out["A"][order], [3000.0, 800.0], rtol=1e-4)
    assert out["shift_ns"] == pytest.approx(0.23, abs=1e-4)
    assert out["scatter"] == pytest.approx(2000.0, rel=1e-3)
    assert out["background"] == 3.0 and out["offset"] == 3.0 and out["reconvolution"]
    assert out["n_params"] == 6 and out["full_curve"].shape == t.shape
    assert out["irf_shifted"].sum() == pytest.approx(1.0)
    # the mean lifetimes and fractions of MultiExpFit apply to the IRF-free decay
    assert out["tau_int"] == pytest.approx((3000 * 0.36 + 800 * 16) / (3000 * 0.6 + 800 * 4),
                                           rel=1e-4)


def test_reconvolution_with_noise_mle_and_fitted_background():
    from kpfm_pl import IRFReconvolutionFit

    rng = np.random.default_rng(9)
    t, irf = _irf_setup()
    truth = IRFReconvolutionFit(irf, np.ones(t.size), 1, 0.1, shift=-0.12, background=4.0)
    counts = rng.poisson(truth.full_model(np.array([2000.0, 2.5]))).astype(float)
    fit = IRFReconvolutionFit(irf, counts, 1, 0.1, fit_background=True,
                              statistic="poisson_mle")
    out = fit.to_dict()
    assert out["tau"][0] == pytest.approx(2.5, abs=4 * out["tau_err"][0])
    assert out["shift_ns"] == pytest.approx(-0.12, abs=0.01)
    assert out["background"] == pytest.approx(4.0, abs=4 * out["background_err"])
    assert fit.diagnostics() == []


def test_reconvolution_periodic_kernel_and_fixed_shift():
    from kpfm_pl import IRFReconvolutionFit

    t, irf = _irf_setup(n=500, dt=0.1)        # 50 ns period, tau 20 ns: incomplete decay
    periodic = IRFReconvolutionFit(irf, np.ones(t.size), 1, 0.1, shift=0.0)
    counts = periodic.full_model(np.array([100.0, 20.0]))
    fit = IRFReconvolutionFit(irf, counts, 1, 0.1, shift=0.0)
    assert fit.parameters()["tau"][0] == pytest.approx(20.0, rel=1e-4)
    assert fit.parameters()["A"][0] == pytest.approx(100.0, rel=1e-4)
    assert fit.to_dict()["fit_shift"] is False and fit._n_params() == 2
    single = IRFReconvolutionFit(irf, np.ones(t.size), 1, 0.1, shift=0.0, periodic=False)
    # without the earlier pulses the bins before the rise are empty
    flat = single.full_model(np.array([100.0, 20.0]))
    assert flat[:40].max() < 1e-6 < counts[:40].min()


def test_reconvolution_accepts_amplitude_lifetime_only_guesses_and_bounds():
    from kpfm_pl import IRFReconvolutionFit

    t, irf = _irf_setup()
    counts = IRFReconvolutionFit(irf, np.ones(t.size), 1, 0.1, shift=0.1).full_model(
        np.array([1000.0, 3.0]))
    fit = IRFReconvolutionFit(irf, counts, 1, 0.1, p0=[np.nan, 3.3],
                              bounds=([0, 2.5], [np.inf, 3.5]))
    assert fit.bounds[0].size == 3                       # shift bounds appended
    assert fit.parameters()["tau"][0] == pytest.approx(3.0, rel=1e-4)
    with pytest.raises(ValueError, match="same lifetime grid"):
        IRFReconvolutionFit(irf[:-1], counts, 1, 0.1)
    narrow = IRFReconvolutionFit(irf, counts, 1, 0.1, bounds=([0, 0.5], [np.inf, 2.0]))
    assert any("τ1" in w and "upper bound" in w for w in narrow.diagnostics())
