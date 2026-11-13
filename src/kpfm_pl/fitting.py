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

__all__ = ["MultiExpFit", "poisson_chi_squared", "poisson_deviance", "fit_warnings",
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
        n = len(params) // 2
        A, tau = params[:n], params[n:2 * n]
        return np.exp(-t[:, None] / tau[None, :]) @ A + self.offset

    def _residuals(self, params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = self.model(t, params) - y
        if self.weights is not None:
            r = r * self.weights
        if self.constrain_sum_A:
            violation = np.sum(params[: len(params) // 2]) - 1.0
            r = np.append(r, self._penalty_weight * max(violation, 0.0))
        return r

    def _jacobian(self, params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(params) // 2
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

    def _fit(self, p0, bounds):
        if bounds is None:
            bounds = self._default_bounds()
        try:
            self.bounds = tuple(
                np.broadcast_to(np.asarray(b, dtype=float), (2 * self.n_exp,)).copy()
                for b in bounds)
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
            n = self.n_exp
            size = np.abs(result.x)
            floor = np.repeat([np.max(size[:n]), np.max(size[n:2 * n])], n)
            tolerance = _MLE_RTOL * np.maximum(size, 1e-6 * floor)
            converged = np.all(np.abs(new.x - result.x) <= tolerance)
            result = new
            if converged:
                break
        return result

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
        n = len(params) // 2
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
        if corrected:
            return p["A"], p["tau"], self.corrected_covariance()
        return p["A_fit"], p["tau"], self.covariance()

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
                            **thresholds)

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


def fit_warnings(A: np.ndarray, tau: np.ndarray, A_err: np.ndarray | None = None,
                 tau_err: np.ndarray | None = None, *,
                 fitted: np.ndarray | None = None,
                 bounds: tuple[np.ndarray, np.ndarray] | None = None,
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
        names = [f"A{i + 1}" for i in range(n)] + [f"τ{i + 1}" for i in range(n)]
        for k, (value, lo, hi) in enumerate(zip(fitted, *bounds)):
            for bound, side in ((lo, "lower"), (hi, "upper")):
                if not np.isfinite(bound) or (k < n and side == "lower" and bound == 0):
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
