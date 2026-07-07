"""Robust multi-exponential decay fitting.

Wraps :func:`scipy.optimize.least_squares` with an analytic Jacobian and an
optional ``sum(A) <= 1`` penalty (handy when fitting normalised histograms).

The model is

.. math::

    f(t) = \\sum_{i=1}^{n} A_i\\, e^{-t/\\tau_i} + C
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import curve_fit, least_squares

__all__ = ["MultiExpFit", "gaussian2d", "fit_gaussian2d", "peak_power_density"]


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
        ``None``.
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
        if p0 is None:
            p0 = self._default_p0()
        if bounds is None:
            bounds = self._default_bounds()
        try:
            return least_squares(
                self._residuals, p0, jac=self._jacobian, args=(self.t, self.y),
                bounds=bounds, loss=self.loss, f_scale=self.f_scale, max_nfev=20000,
            )
        except ValueError as exc:  # pragma: no cover - defensive
            print(f"Multi-exponential fit failed: {exc}")
            return None

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
        params = self.result.x
        n = len(params) // 2
        return {"A": params[:n], "tau": params[n:2 * n]}

    def curve(self, t: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the fitted model (defaults to the fitting grid)."""
        t = self.t if t is None else t
        return self.model(t, self.result.x)

    def quality(self) -> dict[str, Any]:
        """Return goodness-of-fit metrics including ``r_squared``."""
        residuals = self.result.fun
        if self.constrain_sum_A:
            residuals = residuals[:-1]
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        return {
            "residuals": residuals,
            "cost": self.result.cost,
            "success": self.result.success,
            "message": self.result.message,
            "r_squared": 1 - ss_res / ss_tot if ss_tot != 0 else 0.0,
        }

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
        """Bundle parameters, curve and quality into a single dict."""
        params = self.parameters()
        return {
            "A": params["A"],
            "tau": params["tau"],
            "offset": self.offset,
            "curve": self.curve(),
            "quality": self.quality(),
            "label": self.legend_label(),
            "success": self.success,
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
