"""Commodity price models: Schwartz one-factor and Schwartz-Smith two-factor."""

from __future__ import annotations

import numpy as np

from ..core.process import StochasticProcess, check_correlation, check_non_negative, check_positive


class SchwartzOneFactor(StochasticProcess):
    r"""Schwartz (1997) one-factor mean-reverting commodity model

    .. math:: \frac{dS_t}{S_t} = \kappa(\mu - \ln S_t)\,dt + \sigma\,dW_t

    so ``X = ln S`` is an Ornstein-Uhlenbeck process
    :math:`dX_t = \kappa(\alpha - X_t)dt + \sigma dW_t` with
    :math:`\alpha = \mu - \sigma^2 / (2\kappa)`. Simulated exactly.
    Parameters are taken as risk neutral, so :meth:`futures_price` is ``E[S_T]``.
    """

    state_names = ("price",)
    n_factors = 1

    def __init__(self, s0: float, kappa: float, mu: float, sigma: float):
        check_positive(s0=s0, kappa=kappa)
        check_non_negative(sigma=sigma)
        self.s0 = float(s0)
        self.kappa = float(kappa)
        self.mu = float(mu)
        self.sigma = float(sigma)

    @property
    def alpha(self) -> float:
        """Long-run mean of the log price."""
        return self.mu - self.sigma**2 / (2 * self.kappa)

    def _simulate(self, grid, normals, rng):
        n_paths, n_steps, _ = normals.shape
        x = np.empty((n_paths, n_steps + 1))
        x[:, 0] = np.log(self.s0)
        for k, dt in enumerate(grid.dt):
            e = np.exp(-self.kappa * dt)
            sd = self.sigma * np.sqrt(-np.expm1(-2 * self.kappa * dt) / (2 * self.kappa))
            x[:, k + 1] = x[:, k] * e + self.alpha * (1 - e) + sd * normals[:, k, 0]
        return {"price": np.exp(x)}

    def log_mean(self, t):
        """``E[ln S_t]``."""
        e = np.exp(-self.kappa * np.asarray(t, dtype=float))
        return np.log(self.s0) * e + self.alpha * (1 - e)

    def log_variance(self, t):
        """``Var[ln S_t]``."""
        return self.sigma**2 * -np.expm1(-2 * self.kappa * np.asarray(t, dtype=float)) / (2 * self.kappa)

    def futures_price(self, T):
        """Futures price for delivery at ``T``: ``E[S_T]``."""
        return np.exp(self.log_mean(T) + 0.5 * self.log_variance(T))


class SchwartzSmith(StochasticProcess):
    r"""Schwartz-Smith (2000) two-factor commodity model

    .. math::

        \ln S_t = \chi_t + \xi_t,\qquad
        d\chi_t = -\kappa \chi_t\,dt + \sigma_\chi\,dW^\chi_t,\qquad
        d\xi_t = \mu_\xi\,dt + \sigma_\xi\,dW^\xi_t,\qquad
        d\langle W^\chi, W^\xi\rangle_t = \rho\,dt

    The short-term deviation ``chi`` mean-reverts to zero and the equilibrium level
    ``xi`` is a Brownian motion with drift. Both factors are simulated exactly
    from their joint Gaussian transition.

    Parameters
    ----------
    chi0, xi0 : float
        Initial short-term deviation and equilibrium log price.
    """

    state_names = ("price", "chi", "xi")
    n_factors = 2

    def __init__(self, chi0: float, xi0: float, kappa: float, sigma_chi: float,
                 mu_xi: float, sigma_xi: float, rho: float):
        check_positive(kappa=kappa)
        check_non_negative(sigma_chi=sigma_chi, sigma_xi=sigma_xi)
        check_correlation(rho)
        self.chi0 = float(chi0)
        self.xi0 = float(xi0)
        self.kappa = float(kappa)
        self.sigma_chi = float(sigma_chi)
        self.mu_xi = float(mu_xi)
        self.sigma_xi = float(sigma_xi)
        self.rho = float(rho)

    def _step_cov(self, dt: float) -> np.ndarray:
        k, sc, sx = self.kappa, self.sigma_chi, self.sigma_xi
        v_chi = sc**2 * -np.expm1(-2 * k * dt) / (2 * k)
        c = self.rho * sc * sx * -np.expm1(-k * dt) / k
        return np.array([[v_chi, c], [c, sx**2 * dt]])

    def _simulate(self, grid, normals, rng):
        n_paths, n_steps, _ = normals.shape
        chi = np.empty((n_paths, n_steps + 1))
        xi = np.empty((n_paths, n_steps + 1))
        chi[:, 0], xi[:, 0] = self.chi0, self.xi0
        for k, dt in enumerate(grid.dt):
            cov = self._step_cov(dt)
            l11 = np.sqrt(cov[0, 0])
            l21 = cov[1, 0] / l11 if l11 > 0.0 else 0.0
            l22 = np.sqrt(max(cov[1, 1] - l21**2, 0.0))
            shock = normals[:, k, :] @ np.array([[l11, 0.0], [l21, l22]]).T
            chi[:, k + 1] = chi[:, k] * np.exp(-self.kappa * dt) + shock[:, 0]
            xi[:, k + 1] = xi[:, k] + self.mu_xi * dt + shock[:, 1]
        return {"price": np.exp(chi + xi), "chi": chi, "xi": xi}

    def log_mean(self, t):
        """``E[ln S_t]``."""
        t = np.asarray(t, dtype=float)
        return np.exp(-self.kappa * t) * self.chi0 + self.xi0 + self.mu_xi * t

    def log_variance(self, t):
        """``Var[ln S_t]``."""
        t = np.asarray(t, dtype=float)
        k, sc, sx = self.kappa, self.sigma_chi, self.sigma_xi
        return (-np.expm1(-2 * k * t) * sc**2 / (2 * k) + sx**2 * t
                + 2 * -np.expm1(-k * t) * self.rho * sc * sx / k)

    def futures_price(self, T):
        """Futures price ``E[S_T]`` (parameters taken as risk neutral)."""
        return np.exp(self.log_mean(T) + 0.5 * self.log_variance(T))
