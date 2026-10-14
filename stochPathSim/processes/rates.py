"""Short-rate models.

Every model returns the states ``"rate"`` (the short rate) and ``"discount"``
(the pathwise discount factor :math:`\\exp(-\\int_0^t r_s\\,ds)`).

The Gaussian models (:class:`Vasicek`, :class:`HullWhite`, :class:`HoLee` and
:class:`G2pp`) simulate the short rate and its time integral exactly and
jointly, so Monte Carlo bond prices differ from the analytic ones by sampling
error only. They use two standard normals per Brownian driver: one for the
factor and one for its integral, so ``n_factors`` is twice the number of
drivers. The other models integrate the rate with the trapezoid rule.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.process import (
    StochasticProcess,
    check_correlation,
    check_non_negative,
    check_positive,
)
from ..curves import ZeroCurve

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


def _b(k: float, u):
    """``(1 - exp(-k u)) / k``, equal to ``u`` when ``k == 0``."""
    u = np.asarray(u, dtype=float)
    if k == 0.0:
        return u
    return -np.expm1(-k * u) / k


def _integrate(f, upper: float) -> float:
    """Gauss-Legendre integral of a smooth function on ``[0, upper]``."""
    u = 0.5 * upper * (_GL_NODES + 1.0)
    return 0.5 * upper * float(np.sum(_GL_WEIGHTS * f(u)))


def _psd_sqrt(cov: np.ndarray) -> np.ndarray:
    """A matrix ``L`` with ``L @ L.T == cov`` for a positive semi-definite ``cov``."""
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(cov)
        return v * np.sqrt(np.clip(w, 0.0, None))


def _trapezoid_discount(rate: np.ndarray, dt: np.ndarray) -> np.ndarray:
    integral = np.cumsum(0.5 * (rate[:, :-1] + rate[:, 1:]) * dt, axis=1)
    return np.exp(-np.concatenate([np.zeros((rate.shape[0], 1)), integral], axis=1))


def _as_curve(curve) -> ZeroCurve:
    if isinstance(curve, ZeroCurve):
        return curve
    return ZeroCurve.flat(float(curve))


class _GaussianFactors:
    """Exact joint simulation of mean-zero OU factors and their time integrals.

    Factor ``i`` follows ``dx_i = -a_i x_i dt + s_i dW_i`` with
    ``d<W_i, W_j> = corr_ij dt``.
    """

    def __init__(self, speeds, vols, corr):
        self.a = np.asarray(speeds, dtype=float)
        self.s = np.asarray(vols, dtype=float)
        self.corr = np.asarray(corr, dtype=float)
        self.m = self.a.size

    def step_cov(self, dt: float) -> np.ndarray:
        """Covariance of ``(eps_x_1..m, eps_I_1..m)`` over one step of length ``dt``."""
        m = self.m
        kernels = [lambda u, k=k: np.exp(-k * u) for k in self.a] + [
            lambda u, k=k: _b(k, u) for k in self.a
        ]
        vols = np.concatenate([self.s, self.s])
        corr = np.block([[self.corr, self.corr], [self.corr, self.corr]])
        cov = np.empty((2 * m, 2 * m))
        for i in range(2 * m):
            for j in range(i, 2 * m):
                val = corr[i, j] * vols[i] * vols[j] * _integrate(
                    lambda u, i=i, j=j: kernels[i](u) * kernels[j](u), dt)
                cov[i, j] = cov[j, i] = val
        return cov

    def integral_variance(self, t):
        """``Var(int_0^t sum_i x_i ds)`` for factors started at 0."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros_like(t)
        for n, tn in enumerate(t):
            if tn == 0.0:
                continue
            total = 0.0
            for i in range(self.m):
                for j in range(self.m):
                    total += self.corr[i, j] * self.s[i] * self.s[j] * _integrate(
                        lambda u, i=i, j=j: _b(self.a[i], u) * _b(self.a[j], u), tn)
            out[n] = total
        return out

    def convexity_rate(self, t):
        """``0.5 * d/dt Var(int_0^t sum x)``, the drift correction of a fitted model."""
        t = np.asarray(t, dtype=float)
        total = np.zeros_like(t)
        for i in range(self.m):
            for j in range(self.m):
                total = total + self.corr[i, j] * self.s[i] * self.s[j] * _b(self.a[i], t) * _b(self.a[j], t)
        return 0.5 * total

    def simulate(self, x0, dt: np.ndarray, normals: np.ndarray):
        """Return factor paths ``(m, n_paths, n+1)`` and integral paths of the same shape."""
        n_paths, n_steps, _ = normals.shape
        x = np.zeros((self.m, n_paths, n_steps + 1))
        integ = np.zeros_like(x)
        x[:, :, 0] = np.asarray(x0, dtype=float)[:, None]
        chol = {}
        for k in range(n_steps):
            h = float(dt[k])
            key = round(h, 14)
            if key not in chol:
                chol[key] = _psd_sqrt(self.step_cov(h))
            eps = normals[:, k, :] @ chol[key].T
            for i in range(self.m):
                xi = x[i, :, k]
                x[i, :, k + 1] = xi * np.exp(-self.a[i] * h) + eps[:, i]
                integ[i, :, k + 1] = integ[i, :, k] + xi * _b(self.a[i], h) + eps[:, self.m + i]
        return x, integ


class Vasicek(StochasticProcess):
    r"""Vasicek model :math:`dr_t = \kappa(\theta - r_t)\,dt + \sigma\,dW_t`.

    Simulated exactly together with :math:`\int_0^t r_s\,ds`; uses two normals
    per step (``n_factors = 2``).
    """

    state_names = ("rate", "discount")
    n_factors = 2

    def __init__(self, r0: float, kappa: float, theta: float, sigma: float):
        check_positive(kappa=kappa)
        check_non_negative(sigma=sigma)
        self.r0 = float(r0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)

    def _simulate(self, grid, normals, rng):
        factors = _GaussianFactors([self.kappa], [self.sigma], [[1.0]])
        y, integ = factors.simulate([self.r0 - self.theta], grid.dt, normals)
        t = grid.times
        rate = self.theta + y[0]
        discount = np.exp(-(self.theta * t + integ[0]))
        return {"rate": rate, "discount": discount}

    def mean(self, t):
        """Expected short rate ``E[r_t]``."""
        t = np.asarray(t, dtype=float)
        return self.theta + (self.r0 - self.theta) * np.exp(-self.kappa * t)

    def variance(self, t):
        """Variance ``Var[r_t]``."""
        t = np.asarray(t, dtype=float)
        return self.sigma**2 / (2 * self.kappa) * -np.expm1(-2 * self.kappa * t)


class CIR(StochasticProcess):
    r"""Cox-Ingersoll-Ross model :math:`dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t`.

    Parameters
    ----------
    scheme : {"exact", "euler"}
        ``"exact"`` samples the non-central chi-square transition density (it
        draws from the generator, so supplied ``normals`` and antithetic
        sampling do not affect the rate). ``"euler"`` uses the full-truncation
        Euler scheme driven by the supplied normals.
    """

    state_names = ("rate", "discount")
    n_factors = 1

    def __init__(self, r0: float, kappa: float, theta: float, sigma: float, scheme: str = "exact"):
        check_non_negative(r0=r0)
        check_positive(kappa=kappa, theta=theta, sigma=sigma)
        if scheme not in ("exact", "euler"):
            raise ValueError("scheme must be 'exact' or 'euler'")
        self.r0 = float(r0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.scheme = scheme

    @property
    def feller_satisfied(self) -> bool:
        """``2 kappa theta >= sigma^2``: the rate then never reaches zero."""
        return 2 * self.kappa * self.theta >= self.sigma**2

    def _simulate(self, grid, normals, rng):
        n_paths, n_steps, _ = normals.shape
        dt = grid.dt
        rate = np.empty((n_paths, n_steps + 1))
        rate[:, 0] = self.r0
        if self.scheme == "exact":
            df = 4 * self.kappa * self.theta / self.sigma**2
            for k in range(n_steps):
                c = self.sigma**2 * -np.expm1(-self.kappa * dt[k]) / (4 * self.kappa)
                nonc = rate[:, k] * np.exp(-self.kappa * dt[k]) / c
                rate[:, k + 1] = c * rng.noncentral_chisquare(df, nonc)
        else:
            r = rate[:, 0].copy()
            for k in range(n_steps):
                rp = np.maximum(r, 0.0)
                shock = self.sigma * np.sqrt(rp * dt[k]) * normals[:, k, 0]
                r = r + self.kappa * (self.theta - rp) * dt[k] + shock
                rate[:, k + 1] = np.maximum(r, 0.0)
        return {"rate": rate, "discount": _trapezoid_discount(rate, dt)}

    def mean(self, t):
        """Expected short rate ``E[r_t]``."""
        t = np.asarray(t, dtype=float)
        return self.theta + (self.r0 - self.theta) * np.exp(-self.kappa * t)

    def variance(self, t):
        """Variance ``Var[r_t]``."""
        t = np.asarray(t, dtype=float)
        e = np.exp(-self.kappa * t)
        return (self.r0 * self.sigma**2 / self.kappa * (e - e**2)
                + self.theta * self.sigma**2 / (2 * self.kappa) * (1 - e) ** 2)


class _CurveFittedGaussian(StochasticProcess):
    """Short rate ``r = sum_i x_i + shift(t)`` fitted to an initial curve."""

    state_names = ("rate", "discount")

    def _factors(self) -> _GaussianFactors:  # pragma: no cover - abstract
        raise NotImplementedError

    def shift(self, t):
        """Deterministic shift so that ``E[discount(t)] = P(0, t)``."""
        return self.curve.forward_rate(t) + self._factors().convexity_rate(t)

    def _paths(self, grid, normals):
        factors = self._factors()
        x, integ = factors.simulate(np.zeros(factors.m), grid.dt, normals)
        t = grid.times
        shift_integral = -np.log(self.curve.discount(t)) + 0.5 * factors.integral_variance(t)
        rate = x.sum(axis=0) + self.shift(t)
        discount = np.exp(-(integ.sum(axis=0) + shift_integral))
        return x, rate, discount

    @property
    def r0(self) -> float:
        """Initial short rate, the instantaneous forward ``f(0, 0)``."""
        return float(self.shift(0.0))


class HullWhite(_CurveFittedGaussian):
    r"""Hull-White one-factor model :math:`dr_t = (\theta(t) - a r_t)\,dt + \sigma\,dW_t`.

    :math:`\theta(t)` is chosen so that the model reproduces ``curve`` exactly.
    Uses two normals per step (``n_factors = 2``).

    Parameters
    ----------
    curve : ZeroCurve or float
        Initial zero curve, or a flat continuously compounded rate.
    """

    n_factors = 2

    def __init__(self, a: float, sigma: float, curve):
        check_positive(a=a)
        check_non_negative(sigma=sigma)
        self.a = float(a)
        self.sigma = float(sigma)
        self.curve = _as_curve(curve)

    def _factors(self):
        return _GaussianFactors([self.a], [self.sigma], [[1.0]])

    def _simulate(self, grid, normals, rng):
        _, rate, discount = self._paths(grid, normals)
        return {"rate": rate, "discount": discount}


class HoLee(_CurveFittedGaussian):
    r"""Ho-Lee model :math:`dr_t = \theta(t)\,dt + \sigma\,dW_t`, fitted to ``curve``.

    Uses two normals per step (``n_factors = 2``).
    """

    n_factors = 2

    def __init__(self, sigma: float, curve):
        check_non_negative(sigma=sigma)
        self.sigma = float(sigma)
        self.curve = _as_curve(curve)

    def _factors(self):
        return _GaussianFactors([0.0], [self.sigma], [[1.0]])

    def _simulate(self, grid, normals, rng):
        _, rate, discount = self._paths(grid, normals)
        return {"rate": rate, "discount": discount}


class G2pp(_CurveFittedGaussian):
    r"""Two-factor Gaussian model G2++ (Brigo and Mercurio).

    .. math::
        r_t = x_t + y_t + \varphi(t),\quad dx_t = -a x_t\,dt + \sigma\,dW^1_t,\quad
        dy_t = -b y_t\,dt + \eta\,dW^2_t,\quad d\langle W^1, W^2\rangle_t = \rho\,dt

    :math:`\varphi(t)` fits ``curve`` exactly. States: ``"rate"``, ``"x"``,
    ``"y"`` and ``"discount"``. Uses four normals per step (``n_factors = 4``):
    two for the factors and two for their integrals.
    """

    state_names = ("rate", "x", "y", "discount")
    n_factors = 4

    def __init__(self, a: float, b: float, sigma: float, eta: float, rho: float, curve):
        check_positive(a=a, b=b)
        check_non_negative(sigma=sigma, eta=eta)
        check_correlation(rho)
        self.a = float(a)
        self.b = float(b)
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.rho = float(rho)
        self.curve = _as_curve(curve)

    def _factors(self):
        return _GaussianFactors([self.a, self.b], [self.sigma, self.eta],
                                [[1.0, self.rho], [self.rho, 1.0]])

    def _simulate(self, grid, normals, rng):
        x, rate, discount = self._paths(grid, normals)
        return {"rate": rate, "x": x[0], "y": x[1], "discount": discount}


def _theta_function(theta) -> Callable:
    if callable(theta):
        return theta
    value = float(theta)
    return lambda t: np.full_like(np.asarray(t, dtype=float), value)


class BlackKarasinski(StochasticProcess):
    r"""Black-Karasinski model :math:`d\ln r_t = (\theta(t) - a \ln r_t)\,dt + \sigma\,dW_t`.

    ``theta`` is a constant (simulated exactly) or a callable of time (the drift
    integral over each step uses the mid-point value). Fitting ``theta`` exactly
    to a yield curve needs a lattice and is not included.
    """

    state_names = ("rate", "discount")
    n_factors = 1

    def __init__(self, r0: float, a: float, sigma: float, theta=0.0):
        check_positive(r0=r0, a=a)
        check_non_negative(sigma=sigma)
        self.r0 = float(r0)
        self.a = float(a)
        self.sigma = float(sigma)
        self.theta = theta

    def _simulate(self, grid, normals, rng):
        n_paths, n_steps, _ = normals.shape
        theta = _theta_function(self.theta)
        t, dt = grid.times, grid.dt
        z = np.empty((n_paths, n_steps + 1))
        z[:, 0] = np.log(self.r0)
        for k in range(n_steps):
            e = np.exp(-self.a * dt[k])
            drift = float(theta(t[k] + 0.5 * dt[k])) * _b(self.a, dt[k])
            sd = self.sigma * np.sqrt(-np.expm1(-2 * self.a * dt[k]) / (2 * self.a))
            z[:, k + 1] = z[:, k] * e + drift + sd * normals[:, k, 0]
        rate = np.exp(z)
        return {"rate": rate, "discount": _trapezoid_discount(rate, dt)}

    def log_rate_mean(self, t):
        """``E[ln r_t]`` for a constant ``theta``."""
        t = np.asarray(t, dtype=float)
        e = np.exp(-self.a * t)
        return np.log(self.r0) * e + float(self.theta) / self.a * (1 - e)

    def log_rate_variance(self, t):
        """``Var[ln r_t]``."""
        t = np.asarray(t, dtype=float)
        return self.sigma**2 / (2 * self.a) * -np.expm1(-2 * self.a * t)


class BlackDermanToy(StochasticProcess):
    r"""Continuous-time Black-Derman-Toy model with constant volatility.

    .. math:: d\ln r_t = \theta(t)\,dt + \sigma\,dW_t

    With a constant volatility the continuous-time BDT model has no mean
    reversion. ``theta`` is a constant (simulated exactly) or a callable of
    time (mid-point rule over each step).
    """

    state_names = ("rate", "discount")
    n_factors = 1

    def __init__(self, r0: float, sigma: float, theta=0.0):
        check_positive(r0=r0)
        check_non_negative(sigma=sigma)
        self.r0 = float(r0)
        self.sigma = float(sigma)
        self.theta = theta

    def _simulate(self, grid, normals, rng):
        theta = _theta_function(self.theta)
        t, dt = grid.times, grid.dt
        drift = np.asarray(theta(t[:-1] + 0.5 * dt), dtype=float) * dt
        increments = drift + self.sigma * np.sqrt(dt) * normals[:, :, 0]
        z = np.log(self.r0) + np.concatenate(
            [np.zeros((normals.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1)
        rate = np.exp(z)
        return {"rate": rate, "discount": _trapezoid_discount(rate, dt)}


__all__ = ["Vasicek", "CIR", "HullWhite", "HoLee", "G2pp", "BlackKarasinski", "BlackDermanToy"]
