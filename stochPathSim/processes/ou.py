"""Ornstein-Uhlenbeck process."""

from __future__ import annotations

import numpy as np

from ..core.process import StochasticProcess, check_non_negative, check_positive


class OrnsteinUhlenbeck(StochasticProcess):
    r"""Ornstein-Uhlenbeck process

    .. math:: dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t

    simulated exactly from its Gaussian transition density.

    Parameters
    ----------
    x0 : float
        Initial value.
    kappa : float
        Mean-reversion speed (> 0).
    theta : float
        Long-run mean.
    sigma : float
        Volatility.
    """

    state_names = ("value",)
    n_factors = 1

    def __init__(self, x0: float, kappa: float, theta: float, sigma: float):
        check_positive(kappa=kappa)
        check_non_negative(sigma=sigma)
        self.x0 = float(x0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)

    def _simulate(self, grid, normals, rng):
        dt = grid.dt
        decay = np.exp(-self.kappa * dt)
        sd = self.sigma * np.sqrt(-np.expm1(-2.0 * self.kappa * dt) / (2.0 * self.kappa))
        n_paths = normals.shape[0]
        x = np.empty((n_paths, grid.n_steps + 1))
        x[:, 0] = self.x0
        for i in range(grid.n_steps):
            x[:, i + 1] = self.theta + (x[:, i] - self.theta) * decay[i] + sd[i] * normals[:, i, 0]
        return {"value": x}

    def mean(self, t):
        """Expected value ``E[X_t]``."""
        t = np.asarray(t, dtype=float)
        return self.theta + (self.x0 - self.theta) * np.exp(-self.kappa * t)

    def variance(self, t):
        """Variance ``Var[X_t]``."""
        t = np.asarray(t, dtype=float)
        return self.sigma**2 * -np.expm1(-2.0 * self.kappa * t) / (2.0 * self.kappa)
