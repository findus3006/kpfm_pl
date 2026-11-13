"""Robust multi-exponential decay fitting.

Wraps :func:`scipy.optimize.least_squares` with an analytic Jacobian and an
optional ``sum(A) <= 1`` penalty (handy when fitting normalised histograms).

The model is

.. math::

    f(t) = \\sum_{i=1}^{n} A_i\\, e^{-t/\\tau_i} + C

Besides (weighted) least squares, :class:`MultiExpFit` offers a Poisson
maximum-likelihood fit (``statistic="poisson_mle"``) and, for periodic
excitation, the correction of the amplitudes for the incomplete decay of
earlier pulses (``period``).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.optimize import curve_fit, least_squares

__all__ = ["MultiExpFit", "IRFReconvolutionFit", "prepare_irf", "irf_fwhm", "exp_kernel",
           "poisson_chi_squared", "poisson_deviance", "fit_warnings",
           "amplitude_fractions", "intensity_fractions", "mean_lifetime",
           "gaussian2d", "fit_gaussian2d", "peak_power_density"]

#: Fit statistics of :class:`MultiExpFit`.
STATISTICS = ("least_squares", "poisson_mle")
#: Kinds of mean lifetime, see :func:`mean_lifetime`.
MEAN_LIFETIMES = ("intensity", "amplitude")
#: Iterations of the iteratively reweighted least squares behind the Poisson MLE.
_MLE_MAX_ITER = 30
#: Relative parameter change below which the MLE iterations stop.
_MLE_RTOL = 1e-7
#: Smallest expected count used for the MLE weights 1/sqrt(model). Bins with
#: fewer expected counts carry almost no information; the floor keeps their
#: weight finite.
_MLE_MODEL_FLOOR = 0.1


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``numerator / denominator`` with ``NaN`` where the denominator is 0 or not finite."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = numerator / denominator
    out = np.where(np.isfinite(denominator) & (denominator != 0), out, np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def amplitude_fractions(A: np.ndarray) -> np.ndarray:
    """Amplitude fractions ``A_i / sum(A)`` along the last axis.

    The share of excited molecules (or emitters) in each component at the
    excitation time. ``NaN`` where all amplitudes are zero. Works on one fit
    (shape ``(n_exp,)``) or on many (e.g. ``(time bins, n_exp)``).
    """
    A = np.asarray(A, dtype=float)
    return _safe_ratio(A, np.sum(A, axis=-1, keepdims=True))


def intensity_fractions(A: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Intensity fractions ``A_i tau_i / sum(A tau)`` along the last axis.

    The share of the emitted photons in each component (the integral of
    ``A_i exp(-t/tau_i)`` is ``A_i tau_i``). ``NaN`` where the total is zero.
    """
    weight = np.asarray(A, dtype=float) * np.asarray(tau, dtype=float)
    return _safe_ratio(weight, np.sum(weight, axis=-1, keepdims=True))


def mean_lifetime(A: np.ndarray, tau: np.ndarray, kind: str = "intensity") -> np.ndarray:
    """Mean lifetime of a multi-exponential decay along the last axis.

    ``kind="intensity"`` (default): ``tau_int = sum(A tau^2) / sum(A tau)``,
    the mean weighted by the photons of each component, which equals the mean
    photon arrival time after the excitation. ``kind="amplitude"``:
    ``tau_amp = sum(A tau) / sum(A)``, weighted by the amplitudes (share of
    emitters), e.g. for quenching/FRET analyses. Both equal ``tau`` for one
    exponential. ``NaN`` where the weights sum to zero or a value is not
    finite.
    """
    if kind not in MEAN_LIFETIMES:
        raise ValueError(f"kind must be one of {MEAN_LIFETIMES}, not {kind!r}.")
    A = np.asarray(A, dtype=float)
    tau = np.asarray(tau, dtype=float)
    if kind == "intensity":
        return _safe_ratio(np.sum(A * tau ** 2, axis=-1), np.sum(A * tau, axis=-1))
    return _safe_ratio(np.sum(A * tau, axis=-1), np.sum(A, axis=-1))


class MultiExpFit:
    """Fit a sum of ``n_exp`` exponentials (plus offset ``C``) to ``(t, y)``.

    Parameters
    ----------
    t, y :
        Time and intensity data.
    n_exp :
        Number of exponential components.
    offset :
        Constant offset ``C`` added to the model (e.g. dark counts).
    p0 :
        Optional initial guess ``[A_1..A_n, tau_1..tau_n]``. Auto-generated when
        ``None``; individual ``NaN`` entries are auto-generated too. Automatic
        guesses are clipped into ``bounds``, explicit ones are used as given.
    bounds :
        Optional ``(lower, upper)`` arrays. Auto-generated when ``None``.
    loss, f_scale :
        Passed through to :func:`scipy.optimize.least_squares` for robust losses.
    weights :
        Optional per-sample weights for the residuals.
    constrain_sum_A :
        When ``True`` add a penalty enforcing ``sum(A) <= 1``.
    statistic :
        ``"least_squares"`` (default) minimises the (weighted) squared
        residuals. ``"poisson_mle"`` maximises the Poisson likelihood of the
        photon counts ``y * count_scale``, i.e. minimises the Poisson deviance
        (:func:`poisson_deviance`). It is computed by iteratively reweighted
        least squares with weights ``1 / sqrt(model)``, whose fixed point is
        exactly the maximum-likelihood estimate. Least squares underestimates
        lifetimes at low photon numbers; the MLE does not. Requires the linear
        ``loss`` and no explicit ``weights``.
    count_scale :
        Factor converting ``y`` (and the model) into photon counts, e.g. the
        normalisation peak of a normalised histogram. Only used by
        ``statistic="poisson_mle"``.
    period :
        Laser repetition period in the units of ``t``. With periodic
        excitation every bin also holds photons of all earlier pulses; for an
        exponential they add up to ``exp(-t/tau) / (1 - exp(-period/tau))``,
        so the fitted lifetimes are correct but the fitted amplitudes are too
        large. When given, :meth:`parameters`, :meth:`to_dict` and
        :meth:`intensity_weighted_lifetime` report the per-pulse amplitudes
        ``A = A_fit * (1 - exp(-period/tau))``; the fitted ones stay
        available as ``A_fit``. ``None`` (default) applies no correction.
    """

    def __init__(
        self,
        t: np.ndarray,
        y: np.ndarray,
        n_exp: int,
        offset: float = 0.0,
        *,
        p0: np.ndarray | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        loss: str = "linear",
        f_scale: float = 1.0,
        weights: np.ndarray | None = None,
        constrain_sum_A: bool = False,
        statistic: str = "least_squares",
        count_scale: float = 1.0,
        period: float | None = None,
    ) -> None:
        self.t = np.asarray(t, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n_exp = int(n_exp)
        self.offset = float(offset)
        self.loss = loss
        self.f_scale = f_scale
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.constrain_sum_A = bool(constrain_sum_A)
        self._penalty_weight = 1e5 * np.max(np.abs(self.y)) if self.constrain_sum_A else 0.0
        if statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {STATISTICS}, not {statistic!r}.")
        if statistic == "poisson_mle":
            if loss != "linear":
                raise ValueError("statistic='poisson_mle' requires loss='linear'.")
            if weights is not None:
                raise ValueError("statistic='poisson_mle' sets its own weights; do not "
                                 "pass weights.")
        if not np.isfinite(count_scale) or count_scale <= 0:
            raise ValueError("count_scale must be a positive finite number.")
        if period is not None and (not np.isfinite(period) or period <= 0):
            raise ValueError("period must be a positive finite number or None.")
        self.statistic = statistic
        self.count_scale = float(count_scale)
        self.period = None if period is None else float(period)
        self.bounds: tuple[np.ndarray, np.ndarray] | None = None

        self._error: Exception | None = None
        self.result = self._fit(p0=p0, bounds=bounds)

    # -- model -----------------------------------------------------------------
    def model(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Evaluate the multi-exponential model at ``t``."""
        t = np.asarray(t, dtype=float)
        n = self.n_exp
        A, tau = params[:n], params[n:2 * n]
        return np.exp(-t[:, None] / tau[None, :]) @ A + self.offset

    def _residuals(self, params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = self.model(t, params) - y
        if self.weights is not None:
            r = r * self.weights
        if self.constrain_sum_A:
            violation = np.sum(params[:self.n_exp]) - 1.0
            r = np.append(r, self._penalty_weight * max(violation, 0.0))
        return r

    def _jacobian(self, params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = self.n_exp
        A, tau = params[:n], params[n:2 * n]
        exp_terms = np.exp(-t[:, None] / tau[None, :])
        d_tau = (t[:, None] / tau[None, :] ** 2) * A[None, :] * exp_terms
        jac = np.column_stack([exp_terms, d_tau])
        if self.weights is not None:
            jac = jac * self.weights[:, None]
        if self.constrain_sum_A:
            extra = np.zeros((1, jac.shape[1]))
            if np.sum(A) > 1.0:
                extra[0, :n] = self._penalty_weight
            jac = np.vstack([jac, extra])
        return jac

    # -- hooks for subclasses with extra parameters -----------------------------
    def _n_params(self) -> int:
        """Number of fitted parameters (``[A_1..A_n, tau_1..tau_n]`` plus extras)."""
        return 2 * self.n_exp

    def _param_names(self) -> list[str]:
        """Display names of the fitted parameters (used in warnings)."""
        n = self.n_exp
        return [f"A{i + 1}" for i in range(n)] + [f"τ{i + 1}" for i in range(n)]

    def _complete(self, values, defaults: np.ndarray) -> np.ndarray:
        """Guesses or bounds for all parameters.

        A scalar is broadcast; a vector of ``2 * n_exp`` entries (amplitudes and
        lifetimes only) is extended by the ``defaults`` of the extra parameters.
        """
        n2, total = 2 * self.n_exp, self._n_params()
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            return np.full(total, float(values))
        if values.size == total:
            return values.astype(float).copy()
        if values.size == n2:
            return np.concatenate([values, np.asarray(defaults, dtype=float)[n2:]])
        raise ValueError(f"Expected {n2} or {total} values, got {values.size}.")

    def _fit(self, p0, bounds):
        defaults = self._default_bounds()
        if bounds is None:
            bounds = defaults
        try:
            bounds = tuple(self._complete(b, d) for b, d in zip(bounds, defaults))
            self.bounds = bounds
            if p0 is not None:
                p0 = self._complete(p0, np.full(self._n_params(), np.nan))
            p0 = self._initial_guess(p0, bounds)
            if self.statistic == "poisson_mle":
                return self._fit_poisson_mle(p0, bounds)
            return self._least_squares(p0, bounds)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._error = exc
            warnings.warn(
                f"Multi-exponential fit failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _least_squares(self, p0, bounds):
        return least_squares(
            self._residuals, p0, jac=self._jacobian, args=(self.t, self.y),
            bounds=bounds, loss=self.loss, f_scale=self.f_scale, max_nfev=20000,
        )

    def _fit_poisson_mle(self, p0, bounds):
        """Poisson MLE by iteratively reweighted least squares.

        Each pass minimises ``sum((y - m)^2 / m_prev)`` with the model of the
        previous pass fixed in the weights. At the fixed point ``m = m_prev``
        the gradient equals that of the Poisson deviance, so the result is the
        maximum-likelihood estimate (Dembinski, Schmelling & Waldi,
        arXiv:1807.07911). The first pass uses the Neyman weights
        ``1/sqrt(max(counts, 1))``.
        """
        scale = self.count_scale
        self.weights = scale / np.sqrt(np.maximum(self.y * scale, 1.0))
        result = self._least_squares(p0, bounds)
        for _ in range(_MLE_MAX_ITER):
            expected = self.model(self.t, result.x) * scale
            self.weights = scale / np.sqrt(np.maximum(expected, _MLE_MODEL_FLOOR))
            new = self._least_squares(result.x, bounds)
            # Relative change per parameter; amplitudes at 0 are compared with
            # the largest amplitude, lifetimes with the largest lifetime.
            size = np.abs(result.x)
            tolerance = _MLE_RTOL * np.maximum(size, 1e-6 * self._size_floor(size))
            converged = np.all(np.abs(new.x - result.x) <= tolerance)
            result = new
            if converged:
                break
        return result

    def _size_floor(self, size: np.ndarray) -> np.ndarray:
        """Scale of each parameter for the MLE convergence test (zeros allowed)."""
        n = self.n_exp
        return np.repeat([np.max(size[:n]), np.max(size[n:2 * n])], n)

    def _require_result(self):
        """Return the optimizer result or raise a clear error if the fit failed.

        Guards the accessors so a failed fit surfaces as an explicit
        :class:`RuntimeError` (carrying the original cause) instead of an opaque
        ``AttributeError`` from dereferencing ``None``.
        """
        if self.result is None:
            msg = "Multi-exponential fit did not converge; no parameters available."
            if self._error is not None:
                raise RuntimeError(msg) from self._error
            raise RuntimeError(msg)
        return self.result

    def _initial_guess(self, p0, bounds) -> np.ndarray:
        """Fill ``None``/``NaN`` guesses automatically and clip those into ``bounds``.

        Explicit guesses are left untouched, so an infeasible one still makes
        the fit fail instead of being silently moved.
        """
        auto = self._default_p0()
        if p0 is None:
            p0 = np.full(auto.shape, np.nan)
        p0 = np.array(p0, dtype=float)
        missing = np.isnan(p0)
        if not missing.any():
            return p0
        lower, upper = (np.broadcast_to(np.asarray(b, dtype=float), p0.shape) for b in bounds)
        p0[missing] = np.clip(auto, lower, upper)[missing]
        return p0

    def _default_p0(self) -> np.ndarray:
        A_guess = np.full(self.n_exp, np.max(self.y) / self.n_exp)
        t_lower = (self.t[1] - self.t[0]) * 2
        tail_thresh = np.mean(self.y[-30:]) + 2 * np.std(self.y[-30:])
        valid = np.where(self.y > tail_thresh)[0]
        t_upper = self.t[valid[-1]] if len(valid) else self.t.max()
        tau_guess = np.logspace(np.log10(t_lower), np.log10(max(t_upper, t_lower * 2)), self.n_exp)
        return np.concatenate([A_guess, tau_guess])

    def _default_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        upper_A = np.ones(self.n_exp) if self.constrain_sum_A else np.full(self.n_exp, np.inf)
        lower = np.concatenate([np.zeros(self.n_exp), np.full(self.n_exp, 1e-10)])
        upper = np.concatenate([upper_A, np.full(self.n_exp, np.inf)])
        return lower, upper

    # -- accessors -------------------------------------------------------------
    @property
    def success(self) -> bool:
        return bool(self.result is not None and self.result.success)

    def _period_factor(self, tau: np.ndarray) -> np.ndarray:
        """``1 - exp(-period/tau)`` (ones without ``period``): per-pulse / fitted amplitude."""
        tau = np.asarray(tau, dtype=float)
        if self.period is None:
            return np.ones_like(tau)
        return -np.expm1(-self.period / tau)

    def parameters(self) -> dict[str, np.ndarray]:
        """Return amplitudes and lifetimes as ``{"A": ..., "tau": ..., "A_fit": ...}``.

        ``A_fit`` are the fitted amplitudes of the model curve. ``A`` are the
        per-pulse amplitudes, corrected for the incomplete decay of earlier
        pulses when ``period`` is set (otherwise equal to ``A_fit``).
        """
        params = self._require_result().x
        n = self.n_exp
        A_fit, tau = params[:n], params[n:2 * n]
        return {"A": A_fit * self._period_factor(tau), "tau": tau, "A_fit": A_fit}

    def curve(self, t: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the fitted model (defaults to the fitting grid)."""
        t = self.t if t is None else t
        return self.model(t, self._require_result().x)

    def quality(self) -> dict[str, Any]:
        """Return goodness-of-fit metrics including ``r_squared``.

        ``"residuals"`` are the (possibly weighted) residuals as minimised by
        the optimiser, excluding the ``constrain_sum_A`` penalty term.
        ``"r_squared"`` is always computed from the unweighted residuals
        ``curve() - y`` so it stays meaningful with non-unit ``weights``.
        """
        result = self._require_result()
        residuals = result.fun
        if self.constrain_sum_A:
            residuals = residuals[:-1]
        ss_res = np.sum((self.curve() - self.y) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        return {
            "residuals": residuals,
            "cost": result.cost,
            "success": result.success,
            "message": result.message,
            "r_squared": 1 - ss_res / ss_tot if ss_tot != 0 else 0.0,
        }

    def covariance(self) -> np.ndarray:
        """Parameter covariance ``[A_1..A_n, tau_1..tau_n]`` from the fit Jacobian.

        ``(J^T J)^-1`` scaled by the residual variance ``sum(r^2) / dof`` of
        the (possibly weighted) residuals, as :func:`scipy.optimize.curve_fit`
        does by default, so the errors also reflect a poor fit. All ``NaN``
        without degrees of freedom.
        """
        result = self._require_result()
        jac = np.asarray(result.jac, dtype=float)
        residuals = np.asarray(result.fun, dtype=float)
        if self.constrain_sum_A:
            jac, residuals = jac[:-1], residuals[:-1]
        n_params = jac.shape[1]
        dof = residuals.size - n_params
        if dof <= 0:
            return np.full((n_params, n_params), np.nan)
        variance = float(np.sum(residuals ** 2)) / dof
        return np.linalg.pinv(jac.T @ jac) * variance

    def corrected_covariance(self) -> np.ndarray:
        """Covariance of the per-pulse parameters ``[A_1..A_n, tau_1..tau_n]``.

        :meth:`covariance` propagated through ``A = A_fit * (1 - exp(-T/tau))``
        (identical to it without ``period``).
        """
        cov = self.covariance()
        if self.period is None:
            return cov
        p = self.parameters()
        tau, A_fit = np.asarray(p["tau"], dtype=float), np.asarray(p["A_fit"], dtype=float)
        n = self.n_exp
        jac = np.eye(2 * n)
        jac[:n, :n] = np.diag(self._period_factor(tau))
        # d/dtau of (1 - exp(-T/tau)) = -exp(-T/tau) * T / tau^2
        jac[:n, n:] = np.diag(-A_fit * np.exp(-self.period / tau) * self.period / tau ** 2)
        return jac @ cov @ jac.T

    def _values(self, corrected: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(A, tau, covariance)``: per-pulse (``corrected``) or fitted amplitudes."""
        p = self.parameters()
        n2 = 2 * self.n_exp
        if corrected:
            return p["A"], p["tau"], self.corrected_covariance()[:n2, :n2]
        return p["A_fit"], p["tau"], self.covariance()[:n2, :n2]

    @staticmethod
    def _propagate(gradient: np.ndarray, cov: np.ndarray) -> float:
        """Standard error ``sqrt(g^T C g)`` (``NaN`` for a negative variance)."""
        variance = float(gradient @ cov @ gradient)
        return float(np.sqrt(variance)) if variance >= 0 else float("nan")

    def mean_lifetime(self, kind: str = "intensity",
                      corrected: bool = True) -> tuple[float, float]:
        """Mean lifetime (see :func:`mean_lifetime`) and its standard error.

        ``kind`` is ``"intensity"`` (``sum(A tau^2) / sum(A tau)``) or
        ``"amplitude"`` (``sum(A tau) / sum(A)``). Uses the per-pulse
        amplitudes ``A`` (see ``period``); ``corrected=False`` uses the fitted
        ``A_fit`` instead. The error is propagated from the covariance,
        including the correlations of amplitudes and lifetimes; both are
        ``NaN`` when the weights sum to zero.
        """
        A, tau, cov = self._values(corrected)
        A, tau = np.asarray(A, dtype=float), np.asarray(tau, dtype=float)
        mean = float(mean_lifetime(A, tau, kind))
        if not np.isfinite(mean):
            return float("nan"), float("nan")
        if kind == "intensity":
            denominator = float(np.sum(A * tau))
            gradient = np.concatenate([tau * (tau - mean), A * (2 * tau - mean)]) / denominator
        else:
            denominator = float(np.sum(A))
            gradient = np.concatenate([tau - mean, A]) / denominator
        return mean, self._propagate(gradient, cov)

    def intensity_weighted_lifetime(self, corrected: bool = True) -> tuple[float, float]:
        """Intensity-weighted mean lifetime ``sum(A tau^2) / sum(A tau)`` and its error.

        Equal to ``tau`` for one exponential. Shorthand for
        :meth:`mean_lifetime` with ``kind="intensity"``.
        """
        return self.mean_lifetime("intensity", corrected)

    def fractions(self, kind: str = "amplitude",
                  corrected: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Component fractions and their standard errors.

        ``kind="amplitude"``: ``A_i / sum(A)`` (:func:`amplitude_fractions`);
        ``kind="intensity"``: ``A_i tau_i / sum(A tau)``
        (:func:`intensity_fractions`). Fractions of 0..1, ``NaN`` when the
        total is zero. Errors are propagated from the covariance.
        """
        if kind not in MEAN_LIFETIMES:
            raise ValueError(f"kind must be one of {MEAN_LIFETIMES}, not {kind!r}.")
        A, tau, cov = self._values(corrected)
        A, tau = np.asarray(A, dtype=float), np.asarray(tau, dtype=float)
        n = A.size
        nan = np.full(n, np.nan)
        if kind == "amplitude":
            values = amplitude_fractions(A)
            total = float(np.sum(A))
            if not np.all(np.isfinite(values)):
                return nan, nan.copy()
            jac = np.zeros((n, 2 * n))
            jac[:, :n] = (np.eye(n) - values[:, None]) / total
        else:
            values = intensity_fractions(A, tau)
            total = float(np.sum(A * tau))
            if not np.all(np.isfinite(values)):
                return nan, nan.copy()
            jac = np.zeros((n, 2 * n))
            jac[:, :n] = (np.diag(tau) - values[:, None] * tau[None, :]) / total
            jac[:, n:] = (np.diag(A) - values[:, None] * A[None, :]) / total
        errors = np.array([self._propagate(row, cov) for row in jac])
        return values, errors

    def diagnostics(self, **thresholds: float) -> list[str]:
        """Warnings about a probably over-parameterised fit, see :func:`fit_warnings`.

        Checks the per-pulse amplitudes and lifetimes (with their errors) and
        whether a fitted parameter ended at one of the fit's bounds.
        ``thresholds`` are forwarded to :func:`fit_warnings`.
        """
        p = self.parameters()
        errors = np.sqrt(np.clip(np.diag(self.corrected_covariance()), 0.0, None))
        n = self.n_exp
        return fit_warnings(p["A"], p["tau"], errors[:n], errors[n:2 * n],
                            fitted=self._require_result().x, bounds=self.bounds,
                            names=self._param_names(), **thresholds)

    def legend_label(self, precision: int = 2) -> str:
        """Return a LaTeX string describing the fitted model curve for legends.

        Uses the fitted amplitudes ``A_fit``, so the formula reproduces the
        plotted curve.
        """
        p = self.parameters()
        terms = [rf"{a:.{precision}f} e^{{-t/{tau:.{precision}f}\,\mathrm{{ns}}}}"
                 for a, tau in zip(p["A_fit"], p["tau"])]
        label = " + ".join(terms)
        if abs(self.offset) > 1e-12:
            label += rf" + {self.offset:.{precision}f}"
        return r"$f(t) = " + label + r"$"

    def to_dict(self) -> dict[str, Any]:
        """Bundle parameters, curve and quality into a single dict.

        Includes the standard errors ``A_err``/``tau_err`` (from
        :meth:`corrected_covariance`) and the intensity-weighted mean lifetime
        ``tau_int`` with its error ``tau_int_err``
        (:meth:`intensity_weighted_lifetime`), the amplitude-weighted mean
        lifetime ``tau_amp`` with ``tau_amp_err``, and the amplitude and
        intensity fractions ``f_amp``/``f_int`` (0..1) with their errors
        ``f_amp_err``/``f_int_err`` (:meth:`fractions`). These are per-pulse
        values (see ``period``); ``A_fit``/``A_fit_err``/``tau_int_fit``/
        ``tau_amp_fit`` are the uncorrected values of the fitted curve.
        """
        params = self.parameters()
        n = self.n_exp
        errors = np.sqrt(np.clip(np.diag(self.corrected_covariance()), 0.0, None))
        fit_errors = np.sqrt(np.clip(np.diag(self.covariance()), 0.0, None))
        tau_int, tau_int_err = self.mean_lifetime("intensity")
        tau_amp, tau_amp_err = self.mean_lifetime("amplitude")
        f_amp, f_amp_err = self.fractions("amplitude")
        f_int, f_int_err = self.fractions("intensity")
        return {
            "A": params["A"],
            "tau": params["tau"],
            "A_err": errors[:n],
            "tau_err": errors[n:2 * n],
            "tau_int": tau_int,
            "tau_int_err": tau_int_err,
            "tau_amp": tau_amp,
            "tau_amp_err": tau_amp_err,
            "f_amp": f_amp,
            "f_amp_err": f_amp_err,
            "f_int": f_int,
            "f_int_err": f_int_err,
            "A_fit": params["A_fit"],
            "A_fit_err": fit_errors[:n],
            "tau_int_fit": self.mean_lifetime("intensity", corrected=False)[0],
            "tau_amp_fit": self.mean_lifetime("amplitude", corrected=False)[0],
            "period": self.period,
            "statistic": self.statistic,
            "offset": self.offset,
            "curve": self.curve(),
            "quality": self.quality(),
            "label": self.legend_label(),
            "success": self.success,
        }


# ------------------------------------------------------------ IRF reconvolution
def _irf_floor(counts: np.ndarray) -> float:
    """Mean of the floor bins (below ``median + 5 sqrt(median)``) of an IRF histogram."""
    median = float(np.median(counts))
    floor = counts[counts < median + 5.0 * np.sqrt(max(median, 1.0))]
    return float(np.mean(floor)) if floor.size else median


def prepare_irf(counts: np.ndarray, background: float | str | None = "auto",
                ) -> tuple[np.ndarray, float]:
    """Turn a recorded IRF histogram into a unit-area response.

    Subtracts a constant ``background`` and normalises to ``sum = 1``.
    Without the background subtraction a flat IRF floor, convolved with the
    decay, would add a spurious constant (or slow) component to the model.

    ``background="auto"`` estimates the floor as the mean of the floor bins:
    those below ``median + 5 sqrt(median)``, which excludes the IRF peak and
    tail (for a narrow IRF the floor fills most of the period). The mean, not
    the median, is used because the median of Poisson counts is biased.
    ``None`` subtracts nothing, a number subtracts that level.

    The result is **not clipped** at zero: the floor keeps its zero-mean
    noise. Clipping would remove only the negative noise and leave a positive
    floor (about ``0.4 sqrt(floor)`` counts per bin), which the convolution
    turns into a flat level that the fit then takes from the background ``B``
    (4 % of ``B`` on a synthetic test file).

    Returns
    -------
    (irf, background) :
        The normalised IRF and the background that was subtracted.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or counts.size < 3:
        raise ValueError("The IRF must be a 1-D histogram with at least 3 bins.")
    if isinstance(background, str):
        if background != "auto":
            raise ValueError(f"background must be 'auto', None or a number, not {background!r}.")
        level = _irf_floor(counts)
    else:
        level = 0.0 if background is None else float(background)
    irf = counts - level
    total = float(np.sum(irf))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("The IRF contains no counts above its background.")
    return irf / total, level


def irf_fwhm(irf: np.ndarray) -> float:
    """Full width at half maximum of an IRF in bins (circular, linear interpolation)."""
    irf = np.asarray(irf, dtype=float)
    n = irf.size
    peak = int(np.argmax(irf))
    half = irf[peak] / 2.0
    if half <= 0:
        return float("nan")

    def edge(step: int) -> float:
        for k in range(1, n):
            value = irf[(peak + step * k) % n]
            if value < half:
                previous = irf[(peak + step * (k - 1)) % n]
                return k - 1 + (previous - half) / (previous - value)
        return float(n)

    return edge(-1) + edge(1)


def exp_kernel(tau: float, dt: float, n: int, periodic: bool = True) -> np.ndarray:
    """Bin-integrated response of ``exp(-t/tau)`` to an IRF bin, on ``n`` bins.

    For a piecewise-constant IRF and bin-integrated counts, an IRF bin ``j``
    contributes ``k[m - j]`` to bin ``m`` with ``x = dt/tau``, ``q = e^-x``:
    ``k[0] = 1/x - (1 - q)/x^2`` and ``k[d] = q^(d-1) (1 - q)^2 / x^2`` for
    ``d >= 1`` (``sum(k) = tau/dt``; ``k[d] -> q^(d - 1/2)`` for ``tau >> dt``).
    With ``periodic`` the decays of all earlier pulses, ``n`` bins apart, are
    added (geometric series), which is the incomplete-decay correction of the
    reconvolution model.
    """
    x = dt / float(tau)
    one_minus_q = -np.expm1(-x)
    c = (one_minus_q / x) ** 2
    d = np.arange(n, dtype=float)
    k = np.empty(n)
    k[1:] = c * np.exp(-(d[1:] - 1.0) * x)
    k[0] = 1.0 / x - one_minus_q / x ** 2
    if periodic:
        denominator = -np.expm1(-n * x)
        k[1:] /= denominator
        k[0] += c * np.exp(-(n - 1.0) * x) / denominator
    return k


class IRFReconvolutionFit(MultiExpFit):
    """Fit a multi-exponential decay convolved with a measured IRF.

    The model on one laser period of ``N`` bins (width ``dt``) is

    .. math::

        m_j = \\sum_i a_i\\,[g_\\delta \\circledast k_{\\tau_i}]_j + s\\,g_{\\delta,j} + B

    with ``g`` the background-subtracted, unit-area IRF (:func:`prepare_irf`),
    ``g_δ`` the IRF shifted later by ``δ`` (sub-bin, Fourier shift theorem),
    ``k_τ`` the bin-integrated exponential kernel (:func:`exp_kernel`), ``⊛``
    the circular convolution (``periodic``, which includes the decay of all
    earlier pulses) or the linear one, ``s`` the scattered-light counts and
    ``B`` a constant background per bin. ``a_i`` are the amplitudes of the
    IRF-free decay at the excitation time (counts per bin), so the mean
    lifetimes and fractions of :class:`MultiExpFit` apply unchanged.

    The fitted parameters are ``[a_1..a_n, tau_1..tau_n]`` followed by the
    extras ``δ`` (unless ``shift`` is given), ``B`` (with ``fit_background``)
    and ``s`` (with ``scatter``). ``p0`` and ``bounds`` may cover only the
    first ``2 n`` parameters; the extras then use their defaults.

    Parameters
    ----------
    irf :
        Recorded IRF histogram on the same grid as ``counts`` (raw counts).
    counts :
        Decay histogram of one laser period (``N`` bins, any scaling, see
        ``count_scale``).
    n_exp :
        Number of exponential components.
    dt :
        Bin width (ns); all times, lifetimes and the shift are in ns.
    fit_mask :
        Boolean mask of the bins to fit (default: all). Include the rising
        edge and some bins before it.
    t :
        Time of every bin for the output (default ``arange(N) * dt``).
    periodic :
        Circular convolution with the periodic kernel (default) or a linear
        convolution with the single-pulse kernel.
    background :
        Constant background per bin; fixed unless ``fit_background``, then
        the initial value.
    fit_background, scatter :
        Also fit ``B`` / the scatter contribution ``s``.
    scatter_counts :
        Fixed scatter contribution ``s`` when ``scatter`` is off (default 0),
        e.g. taken over from a fit of the summed histogram.
    shift :
        Fixed IRF shift ``δ`` (ns), e.g. from a previous fit of the summed
        histogram; ``None`` fits it.
    shift_guess, shift_range :
        Initial value and half-width of the bounds of a fitted shift (ns).
        By default the guess aligns the half-maximum rising edges of the IRF
        and the decay, and the range is ``max(20 dt, 5 FWHM)`` (at most a
        quarter period).
    irf_background :
        Passed to :func:`prepare_irf`.
    p0, bounds, loss, f_scale, weights, constrain_sum_A, statistic, count_scale :
        As for :class:`MultiExpFit` (``weights`` over the fitted bins).
    """

    def __init__(self, irf: np.ndarray, counts: np.ndarray, n_exp: int, dt: float, *,
                 fit_mask: np.ndarray | None = None, t: np.ndarray | None = None,
                 periodic: bool = True, background: float = 0.0,
                 fit_background: bool = False, scatter: bool = False,
                 scatter_counts: float = 0.0, shift: float | None = None, shift_guess: float | None = None,
                 shift_range: float | None = None,
                 irf_background: float | str | None = "auto",
                 p0: np.ndarray | None = None,
                 bounds: tuple[np.ndarray, np.ndarray] | None = None,
                 loss: str = "linear", f_scale: float = 1.0,
                 weights: np.ndarray | None = None, constrain_sum_A: bool = False,
                 statistic: str = "least_squares", count_scale: float = 1.0) -> None:
        counts = np.asarray(counts, dtype=float)
        n = counts.size
        if np.size(irf) != n:
            raise ValueError(f"The IRF has {np.size(irf)} bins but the decay {n}; both must "
                             "cover the same lifetime grid.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite number.")
        mask = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
        if mask.shape != (n,) or mask.sum() < 2:
            raise ValueError("fit_mask must select at least two bins of the decay.")
        self.counts = counts
        self.dt = float(dt)
        self.n_grid = n
        self.fit_mask = mask
        self.periodic = bool(periodic)
        self.irf, self.irf_background = prepare_irf(irf, irf_background)
        self.irf_fwhm_ns = irf_fwhm(self.irf) * self.dt
        self.fit_shift = shift is None
        self.fixed_shift = None if shift is None else float(shift)
        self.fit_background = bool(fit_background)
        self.fit_scatter = bool(scatter)
        self._background0 = float(background)
        self._scatter0 = float(scatter_counts)
        self._n_fft = n if self.periodic else 2 * n
        self._freq = np.fft.rfftfreq(n)
        self._irf_spectrum = np.fft.rfft(self.irf)
        self._cache: dict[str, Any] = {}
        guess = self._default_shift() if shift_guess is None else float(shift_guess)
        span = shift_range if shift_range is not None else min(
            max(20 * self.dt, 5 * np.nan_to_num(self.irf_fwhm_ns)), n * self.dt / 4)
        self._shift0, self._shift_span = guess, float(span)
        grid_t = np.arange(n) * self.dt if t is None else np.asarray(t, dtype=float)
        if grid_t.shape != (n,):
            raise ValueError("t must have one entry per bin.")
        super().__init__(grid_t[mask], counts[mask], n_exp, offset=self._background0,
                         p0=p0, bounds=bounds, loss=loss, f_scale=f_scale, weights=weights,
                         constrain_sum_A=constrain_sum_A, statistic=statistic,
                         count_scale=count_scale, period=None)

    # -- parameter layout ----------------------------------------------------------
    def _extras(self) -> list[str]:
        return ([*(["shift"] if self.fit_shift else []),
                 *(["background"] if self.fit_background else []),
                 *(["scatter"] if self.fit_scatter else [])])

    def _n_params(self) -> int:
        return 2 * self.n_exp + len(self._extras())

    def _param_names(self) -> list[str]:
        names = {"shift": "IRF shift", "background": "background B", "scatter": "scatter s"}
        return super()._param_names() + [names[e] for e in self._extras()]

    def _unpack(self, params: np.ndarray) -> dict[str, Any]:
        n = self.n_exp
        out = {"A": np.asarray(params[:n], dtype=float),
               "tau": np.asarray(params[n:2 * n], dtype=float),
               "shift": self.fixed_shift, "background": self._background0,
               "scatter": self._scatter0}
        for name, value in zip(self._extras(), params[2 * n:]):
            out[name] = float(value)
        return out

    # -- model -------------------------------------------------------------------
    def _shifted_irf(self, shift_ns: float) -> np.ndarray:
        phase = np.exp(-2j * np.pi * self._freq * (shift_ns / self.dt))
        return np.fft.irfft(self._irf_spectrum * phase, self.n_grid)

    def _irf_derivative(self, shift_ns: float) -> np.ndarray:
        """d g_δ / d δ (per ns)."""
        phase = np.exp(-2j * np.pi * self._freq * (shift_ns / self.dt))
        factor = -2j * np.pi * self._freq / self.dt
        return np.fft.irfft(self._irf_spectrum * phase * factor, self.n_grid)

    def _convolve(self, signal_spectrum: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return np.fft.irfft(signal_spectrum * np.fft.rfft(kernel, self._n_fft),
                            self._n_fft)[:self.n_grid]

    def _kernel(self, tau: float) -> np.ndarray:
        return exp_kernel(tau, self.dt, self.n_grid, self.periodic)

    def _evaluate(self, params: np.ndarray) -> dict[str, Any]:
        """Model on the full grid plus the pieces the Jacobian needs (cached)."""
        key = np.asarray(params, dtype=float).tobytes()
        if self._cache.get("key") == key:
            return self._cache
        p = self._unpack(params)
        g = self._shifted_irf(p["shift"])
        spectrum = np.fft.rfft(g, self._n_fft)
        components = np.array([self._convolve(spectrum, self._kernel(tau)) for tau in p["tau"]])
        full = p["A"] @ components + p["scatter"] * g + p["background"]
        self._cache = {"key": key, "p": p, "g": g, "spectrum": spectrum,
                       "components": components, "full": full}
        return self._cache

    def full_model(self, params: np.ndarray | None = None) -> np.ndarray:
        """Model on the whole period (default: fitted parameters)."""
        params = self._require_result().x if params is None else params
        return self._evaluate(params)["full"].copy()

    def model(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Model on the fitted bins (``t`` must be the fit grid; it is not resampled)."""
        return self._evaluate(params)["full"][self.fit_mask]

    def _jacobian(self, params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        ev = self._evaluate(params)
        p, mask = ev["p"], self.fit_mask
        columns = [c[mask] for c in ev["components"]]
        for a, tau in zip(p["A"], p["tau"]):
            h = 1e-6 * tau
            dk = (self._kernel(tau + h) - self._kernel(tau - h)) / (2 * h)
            columns.append(a * self._convolve(ev["spectrum"], dk)[mask])
        for name in self._extras():
            if name == "shift":
                dg = self._irf_derivative(p["shift"])
                total = np.fft.irfft(
                    np.fft.rfft(dg, self._n_fft)
                    * sum(a * np.fft.rfft(self._kernel(tau), self._n_fft)
                          for a, tau in zip(p["A"], p["tau"])), self._n_fft)[:self.n_grid]
                columns.append((total + p["scatter"] * dg)[mask])
            elif name == "background":
                columns.append(np.ones(int(mask.sum())))
            else:
                columns.append(ev["g"][mask])
        jac = np.column_stack(columns)
        if self.weights is not None:
            jac = jac * self.weights[:, None]
        if self.constrain_sum_A:
            extra = np.zeros((1, jac.shape[1]))
            if np.sum(p["A"]) > 1.0:
                extra[0, :self.n_exp] = self._penalty_weight
            jac = np.vstack([jac, extra])
        return jac

    # -- guesses and bounds ---------------------------------------------------------
    def _default_shift(self) -> float:
        """Shift that aligns the half-maximum rising edges of the IRF and the decay (ns)."""
        n = self.n_grid

        def rise(curve: np.ndarray) -> float:
            floor = float(np.median(curve))
            peak = int(np.argmax(curve))
            half = floor + (curve[peak] - floor) / 2.0
            for k in range(1, n):
                value = curve[(peak - k) % n]
                if value < half:
                    above = curve[(peak - k + 1) % n]
                    return peak - k + (half - value) / (above - value)
            return float(peak)

        smooth = np.convolve(np.concatenate([self.counts[-2:], self.counts, self.counts[:2]]),
                             np.ones(5) / 5, mode="valid")
        delta = (rise(smooth) - rise(self.irf)) % n
        if delta >= n / 2:
            delta -= n
        return float(delta * self.dt)

    def _default_p0(self) -> np.ndarray:
        n = self.n_exp
        background = self._background0 if not self.fit_background else max(
            self._background0, float(np.percentile(self.counts, 5)))
        signal = np.clip(self.counts - background, 0.0, None)
        peak = float(np.max(signal)) if np.max(signal) > 0 else 1.0
        tau0 = max(float(np.sum(signal)) / peak * self.dt, self.dt)
        taus = tau0 * (10.0 ** np.linspace(-1.0, 0.3, n) if n > 1 else np.ones(1))
        amps = float(np.sum(signal)) * self.dt / (n * taus)
        extras = {"shift": self._shift0, "background": background,
                  "scatter": max(self._scatter0, 0.0)}
        return np.concatenate([amps, taus, [extras[e] for e in self._extras()]])

    def _default_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_exp
        upper_A = np.ones(n) if self.constrain_sum_A else np.full(n, np.inf)
        lower = [np.zeros(n), np.full(n, self.dt / 20)]
        upper = [upper_A, np.full(n, 100 * self.n_grid * self.dt)]
        for name in self._extras():
            if name == "shift":
                lower.append([self._shift0 - self._shift_span])
                upper.append([self._shift0 + self._shift_span])
            else:
                lower.append([0.0])
                upper.append([np.inf])
        return np.concatenate(lower), np.concatenate(upper)

    def _size_floor(self, size: np.ndarray) -> np.ndarray:
        # Absolute resolution of the extras (the base test uses 1e-6 * floor).
        scale = {"shift": self.dt, "background": max(float(np.max(np.abs(self.y))), 1.0) * 1e-3,
                 "scatter": max(float(np.sum(np.abs(self.y))), 1.0) * 1e-6}
        return np.concatenate([super()._size_floor(size),
                               [scale[e] / 1e-6 for e in self._extras()]])

    # -- results ---------------------------------------------------------------------
    def extras(self) -> dict[str, tuple[float, float]]:
        """``{"shift_ns", "background", "scatter"}`` → ``(value, error)``; fixed ones have error 0."""
        params = self._require_result().x
        p = self._unpack(params)
        errors = np.sqrt(np.clip(np.diag(self.covariance()), 0.0, None))
        fitted = dict(zip(self._extras(), errors[2 * self.n_exp:]))
        return {"shift_ns": (float(p["shift"]), float(fitted.get("shift", 0.0))),
                "background": (float(p["background"]), float(fitted.get("background", 0.0))),
                "scatter": (float(p["scatter"]), float(fitted.get("scatter", 0.0)))}

    def legend_label(self, precision: int = 2) -> str:
        """LaTeX legend string ``IRF ⊗ Σ a e^{-t/τ}`` of the fitted model."""
        p = self.parameters()
        terms = [rf"{a:.{precision}g} e^{{-t/{tau:.{precision}f}\,\mathrm{{ns}}}}"
                 for a, tau in zip(p["A"], p["tau"])]
        return r"$\mathrm{IRF} \otimes (" + " + ".join(terms) + r")$"

    def to_dict(self) -> dict[str, Any]:
        """:meth:`MultiExpFit.to_dict` plus the reconvolution results.

        Adds ``shift_ns``/``shift_err_ns`` (IRF shift δ, positive = IRF moved
        later), ``background``/``background_err`` (``B`` per bin, also as
        ``offset``), ``scatter``/``scatter_err`` (scattered-light counts),
        ``fit_shift``/``fit_background``/``fit_scatter``, ``periodic``,
        ``irf_background`` (subtracted from the IRF), ``irf_fwhm_ns``,
        ``irf_shifted`` (unit-area IRF after the shift, full grid),
        ``full_curve`` (model on the full grid), ``n_params`` and
        ``reconvolution: True``.
        """
        out = super().to_dict()
        extras = self.extras()
        out.update({
            "shift_ns": extras["shift_ns"][0], "shift_err_ns": extras["shift_ns"][1],
            "background": extras["background"][0], "background_err": extras["background"][1],
            "scatter": extras["scatter"][0], "scatter_err": extras["scatter"][1],
            "offset": extras["background"][0],
            "fit_shift": self.fit_shift, "fit_background": self.fit_background,
            "fit_scatter": self.fit_scatter, "periodic": self.periodic,
            "irf_background": self.irf_background, "irf_fwhm_ns": self.irf_fwhm_ns,
            "irf_shifted": self._shifted_irf(extras["shift_ns"][0]),
            "full_curve": self.full_model(), "n_params": self._n_params(),
            "reconvolution": True,
        })
        return out


def fit_warnings(A: np.ndarray, tau: np.ndarray, A_err: np.ndarray | None = None,
                 tau_err: np.ndarray | None = None, *,
                 fitted: np.ndarray | None = None,
                 bounds: tuple[np.ndarray, np.ndarray] | None = None,
                 names: list[str] | None = None,
                 amp_sigma: float = 2.0, min_fraction: float = 0.01,
                 merge_sigma: float = 2.0, merge_rel: float = 0.1,
                 bound_rel: float = 0.01) -> list[str]:
    """Signs that a multi-exponential fit has more components than the data support.

    Complements a model comparison by BIC: an extra component that the data
    do not need often ends with a vanishing amplitude, a lifetime equal to
    another one, or a parameter stuck at a bound.

    Parameters
    ----------
    A, tau :
        Amplitudes and lifetimes, one entry per component.
    A_err, tau_err :
        Their standard errors (``NaN`` entries skip the error-based checks).
    fitted, bounds :
        The fitted parameter vector ``[A_fit_1..A_fit_n, tau_1..tau_n]`` and
        the fit's ``(lower, upper)`` bounds for the bound check (skipped when
        either is ``None``). Amplitude lower bounds of 0 are skipped: a zero
        amplitude is reported as a collapsed component instead.
    names :
        Display names of the entries of ``fitted`` (default ``A1.., τ1..``;
        needed when ``fitted`` holds extra parameters).
    amp_sigma :
        An amplitude below ``amp_sigma`` times its error is "compatible with
        zero".
    min_fraction :
        A component with an intensity fraction ``A tau / sum(A tau)`` below
        this collapsed.
    merge_sigma, merge_rel :
        Two lifetimes merged if they differ by less than ``merge_sigma``
        combined standard errors or by less than ``merge_rel`` relative to the
        larger one.
    bound_rel :
        A parameter lies at a bound if it is within ``bound_rel`` (relative to
        the bound) of it.

    Returns
    -------
    list of str
        Human-readable warnings, empty when nothing suspicious was found.
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    n = A.size
    A_err = np.full(n, np.nan) if A_err is None else np.atleast_1d(np.asarray(A_err, float))
    tau_err = (np.full(n, np.nan) if tau_err is None
               else np.atleast_1d(np.asarray(tau_err, float)))
    out: list[str] = []

    f_int = intensity_fractions(A, tau)
    for i in range(n):
        name = f"component {i + 1}"
        if np.isfinite(A_err[i]) and A_err[i] > 0 and abs(A[i]) < amp_sigma * A_err[i]:
            out.append(f"Amplitude A{i + 1} = {A[i]:.3g} ± {A_err[i]:.2g} is compatible with "
                       f"zero (< {amp_sigma:g} σ): {name} is probably not needed.")
        elif np.isfinite(f_int[i]) and f_int[i] < min_fraction:
            out.append(f"Component {i + 1} carries only {100 * f_int[i]:.2g} % of "
                       f"the intensity (< {100 * min_fraction:g} %): it has collapsed.")
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(tau[i] - tau[j])
            sigma = float(np.hypot(tau_err[i], tau_err[j]))
            larger = max(abs(tau[i]), abs(tau[j]))
            if (np.isfinite(sigma) and sigma > 0 and diff < merge_sigma * sigma) or \
                    (larger > 0 and diff / larger < merge_rel):
                out.append(f"Lifetimes τ{i + 1} = {tau[i]:.3g} ns and τ{j + 1} = "
                           f"{tau[j]:.3g} ns have merged: they describe the same decay.")
    if fitted is not None and bounds is not None:
        fitted = np.asarray(fitted, dtype=float)
        if names is None:
            names = [f"A{i + 1}" for i in range(n)] + [f"τ{i + 1}" for i in range(n)]
        for k, (value, lo, hi) in enumerate(zip(fitted, *bounds)):
            if k >= len(names):
                break
            for bound, side in ((lo, "lower"), (hi, "upper")):
                # A lower bound of 0 is the physical limit (amplitude, background,
                # scatter), not a constraint the fit fights against.
                if not np.isfinite(bound) or (side == "lower" and bound == 0):
                    continue
                if abs(value - bound) <= bound_rel * max(abs(bound), 1e-300):
                    out.append(f"{names[k]} = {value:.4g} lies at its {side} bound "
                               f"({bound:.4g}): the fit wants to go beyond it.")
    return out


def poisson_deviance(counts: np.ndarray, model: np.ndarray) -> float:
    """Poisson deviance ``2 sum(model - counts + counts ln(counts/model))``.

    ``-2 ln`` of the likelihood ratio between the model and a perfect fit
    (Baker & Cousins 1984). Minimised by the Poisson maximum-likelihood fit
    and approximately chi-squared distributed with ``n - k`` degrees of
    freedom. Model values are floored at a tiny positive number.
    """
    counts = np.asarray(counts, dtype=float)
    model = np.clip(np.asarray(model, dtype=float), 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(counts > 0, counts * np.log(counts / model), 0.0)
    return float(2.0 * np.sum(model - counts + log_term))


#: Statistics the BIC of :func:`poisson_chi_squared` can be based on.
BIC_STATISTICS = ("chi_squared", "deviance")


def poisson_chi_squared(counts: np.ndarray, model: np.ndarray,
                        n_params: int, *, bic_from: str = "chi_squared") -> dict[str, Any]:
    """Goodness of fit of a model to photon counts (Poisson statistics).

    Uses the usual TCSPC weighting ``sigma_i = sqrt(max(counts_i, 1))``, so
    each weighted residual ``(counts_i - model_i) / sigma_i`` has unit
    variance for a correct model. A reduced chi-squared close to 1 indicates a
    good fit; clearly larger values mean systematic deviations (e.g. too few
    exponentials or a badly placed fit window).

    Also returns the Bayesian information criterion
    ``BIC = statistic + n_params * ln(n_points)`` (Schwarz 1978), where the
    statistic is the chi-squared (``bic_from="chi_squared"``, for least-squares
    fits) or the Poisson deviance (``"deviance"``, for maximum-likelihood
    fits); both equal ``-2 ln L`` up to a constant that depends on the data
    only. BIC values are therefore only comparable between fits of the
    *same* counts with the same statistic; the lower BIC is preferred.

    Parameters
    ----------
    counts :
        Measured photon counts per bin (not normalised).
    model :
        Fitted model on the same bins, in counts.
    n_params :
        Number of free fit parameters (``2 * n_exp`` for :class:`MultiExpFit`).
    bic_from :
        ``"chi_squared"`` (default) or ``"deviance"``.

    Returns
    -------
    dict
        ``weighted_residuals`` (array), ``chi_squared``, ``dof`` (degrees of
        freedom ``len(counts) - n_params``), ``reduced_chi_squared`` (``NaN``
        without degrees of freedom), ``deviance``, ``bic``, ``bic_statistic``,
        ``n_points`` and ``n_params``.
    """
    if bic_from not in BIC_STATISTICS:
        raise ValueError(f"bic_from must be one of {BIC_STATISTICS}, not {bic_from!r}.")
    counts = np.asarray(counts, dtype=float)
    model = np.asarray(model, dtype=float)
    if counts.shape != model.shape:
        raise ValueError("counts and model must have the same shape.")
    residuals = (counts - model) / np.sqrt(np.maximum(counts, 1.0))
    chi2 = float(np.sum(residuals ** 2))
    deviance = poisson_deviance(counts, model)
    dof = int(counts.size - n_params)
    statistic = chi2 if bic_from == "chi_squared" else deviance
    bic = statistic + n_params * np.log(counts.size) if counts.size else float("nan")
    return {
        "weighted_residuals": residuals,
        "chi_squared": chi2,
        "dof": dof,
        "reduced_chi_squared": chi2 / dof if dof > 0 else float("nan"),
        "deviance": deviance,
        "bic": float(bic),
        "bic_statistic": bic_from,
        "n_points": int(counts.size),
        "n_params": int(n_params),
    }


def gaussian2d(xy: tuple[np.ndarray, np.ndarray], amplitude: float, x0: float, y0: float,
               sigma_x: float, sigma_y: float) -> np.ndarray:
    """Evaluate an un-rotated 2-D Gaussian (no offset) on a mesh.

    ``xy`` is ``(x, y)`` meshgrid arrays; the result is *raveled* so the
    function can be used directly as a :func:`scipy.optimize.curve_fit` model.
    """
    x, y = xy
    g = amplitude * np.exp(-(((x - x0) ** 2 / (2 * sigma_x ** 2)) +
                             ((y - y0) ** 2 / (2 * sigma_y ** 2))))
    return g.ravel()


def fit_gaussian2d(x: np.ndarray, y: np.ndarray, image: np.ndarray,
                    p0: np.ndarray | None = None, **kwargs: Any):
    """Fit :func:`gaussian2d` to ``image`` sampled on the ``(x, y)`` mesh.

    Parameters
    ----------
    x, y :
        Meshgrid coordinate arrays, same shape as ``image``.
    image :
        2-D intensity data to fit.
    p0 :
        Initial guess ``[A, x0, y0, sigma_x, sigma_y]``. A default is derived
        from ``image``/``x``/``y`` when ``None``.
    **kwargs :
        Extra keyword arguments forwarded to :func:`scipy.optimize.curve_fit`
        (e.g. ``maxfev``).

    Returns
    -------
    (popt, pcov, fit, perr) :
        Optimal parameters, their covariance matrix, the fitted surface
        reshaped like ``image``, and the parameter standard errors
        (``sqrt(diag(pcov))``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    image = np.asarray(image, dtype=float)

    if p0 is None:
        p0 = [np.max(image), np.mean(x), np.mean(y),
              (x.max() - x.min()) / 10, (y.max() - y.min()) / 10]
    kwargs.setdefault("maxfev", 12000)

    popt, pcov = curve_fit(gaussian2d, (x, y), image.ravel(), p0=p0, **kwargs)
    fit = gaussian2d((x, y), *popt).reshape(image.shape)
    perr = np.sqrt(np.diag(pcov))
    return popt, pcov, fit, perr


def peak_power_density(total_power: float, sigma_x: float, sigma_y: float):
    """Peak (center) power density of a 2-D Gaussian beam.

    ``I0 = P / (2*pi*sigma_x*sigma_y)``, equivalent to ``2*P/(pi*w_x*w_y)``
    with ``w = 2*sigma``.
    """
    return total_power / (2 * np.pi * sigma_x * sigma_y)
