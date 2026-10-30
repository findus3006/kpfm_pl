"""Robust multi-exponential decay fitting.

Wraps :func:`scipy.optimize.least_squares` with an analytic Jacobian and an
optional ``sum(A) <= 1`` penalty (handy when fitting normalised histograms).

The model is

.. math::

    f(t) = \\sum_{i=1}^{n} A_i\\, e^{-t/\\tau_i} + C
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.optimize import curve_fit, least_squares

__all__ = ["MultiExpFit", "poisson_chi_squared", "gaussian2d", "fit_gaussian2d",
           "peak_power_density"]


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
            p0 = self._initial_guess(p0, bounds)
            return least_squares(
                self._residuals, p0, jac=self._jacobian, args=(self.t, self.y),
                bounds=bounds, loss=self.loss, f_scale=self.f_scale, max_nfev=20000,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._error = exc
            warnings.warn(
                f"Multi-exponential fit failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

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

    def parameters(self) -> dict[str, np.ndarray]:
        """Return fitted amplitudes and lifetimes as ``{"A": ..., "tau": ...}``."""
        params = self._require_result().x
        n = len(params) // 2
        return {"A": params[:n], "tau": params[n:2 * n]}

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

    def intensity_weighted_lifetime(self) -> tuple[float, float]:
        """Intensity-weighted mean lifetime ``sum(A tau^2) / sum(A tau)`` and its error.

        Equal to ``tau`` for one exponential. The standard error is propagated
        from :meth:`covariance`; both are ``NaN`` when all amplitudes are zero.
        """
        p = self.parameters()
        A, tau = np.asarray(p["A"], dtype=float), np.asarray(p["tau"], dtype=float)
        denominator = float(np.sum(A * tau))
        if denominator == 0 or not np.isfinite(denominator):
            return float("nan"), float("nan")
        mean = float(np.sum(A * tau ** 2)) / denominator
        gradient = np.concatenate([tau * (tau - mean), A * (2 * tau - mean)]) / denominator
        variance = float(gradient @ self.covariance() @ gradient)
        return mean, float(np.sqrt(variance)) if variance >= 0 else float("nan")

    def legend_label(self, precision: int = 2) -> str:
        """Return a LaTeX string describing the fitted model for legends."""
        p = self.parameters()
        terms = [rf"{a:.{precision}f} e^{{-t/{tau:.{precision}f}\,\mathrm{{ns}}}}"
                 for a, tau in zip(p["A"], p["tau"])]
        label = " + ".join(terms)
        if abs(self.offset) > 1e-12:
            label += rf" + {self.offset:.{precision}f}"
        return r"$f(t) = " + label + r"$"

    def to_dict(self) -> dict[str, Any]:
        """Bundle parameters, curve and quality into a single dict.

        Includes the standard errors ``A_err``/``tau_err`` (from
        :meth:`covariance`) and the intensity-weighted mean lifetime
        ``tau_int`` with its error ``tau_int_err``
        (:meth:`intensity_weighted_lifetime`).
        """
        params = self.parameters()
        errors = np.sqrt(np.clip(np.diag(self.covariance()), 0.0, None))
        tau_int, tau_int_err = self.intensity_weighted_lifetime()
        return {
            "A": params["A"],
            "tau": params["tau"],
            "A_err": errors[:self.n_exp],
            "tau_err": errors[self.n_exp:2 * self.n_exp],
            "tau_int": tau_int,
            "tau_int_err": tau_int_err,
            "offset": self.offset,
            "curve": self.curve(),
            "quality": self.quality(),
            "label": self.legend_label(),
            "success": self.success,
        }


def poisson_chi_squared(counts: np.ndarray, model: np.ndarray,
                        n_params: int) -> dict[str, Any]:
    """Goodness of fit of a model to photon counts (Poisson statistics).

    Uses the usual TCSPC weighting ``sigma_i = sqrt(max(counts_i, 1))``, so
    each weighted residual ``(counts_i - model_i) / sigma_i`` has unit
    variance for a correct model. A reduced chi-squared close to 1 indicates a
    good fit; clearly larger values mean systematic deviations (e.g. too few
    exponentials or a badly placed fit window).

    Parameters
    ----------
    counts :
        Measured photon counts per bin (not normalised).
    model :
        Fitted model on the same bins, in counts.
    n_params :
        Number of free fit parameters (``2 * n_exp`` for :class:`MultiExpFit`).

    Returns
    -------
    dict
        ``weighted_residuals`` (array), ``chi_squared``, ``dof`` (degrees of
        freedom ``len(counts) - n_params``) and ``reduced_chi_squared``
        (``NaN`` without degrees of freedom).
    """
    counts = np.asarray(counts, dtype=float)
    model = np.asarray(model, dtype=float)
    if counts.shape != model.shape:
        raise ValueError("counts and model must have the same shape.")
    residuals = (counts - model) / np.sqrt(np.maximum(counts, 1.0))
    chi2 = float(np.sum(residuals ** 2))
    dof = int(counts.size - n_params)
    return {
        "weighted_residuals": residuals,
        "chi_squared": chi2,
        "dof": dof,
        "reduced_chi_squared": chi2 / dof if dof > 0 else float("nan"),
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
