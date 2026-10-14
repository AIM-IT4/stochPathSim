"""Geometric Brownian motion."""

from __future__ import annotations

import numpy as np

from ..core.process import StochasticProcess, check_non_negative, check_positive


class GBM(StochasticProcess):
    r"""Geometric Brownian motion

    .. math:: dS_t = (\mu - q) S_t\,dt + \sigma S_t\,dW_t

    simulated exactly on any grid, so there is no discretisation error.

    Parameters
    ----------
    s0 : float
        Initial value.
    mu : float
        Drift (use the risk-free rate for risk-neutral pricing).
    sigma : float
        Volatility.
    q : float
        Continuous dividend yield.
    """

    state_names = ("spot",)
    n_factors = 1

    def __init__(self, s0: float, mu: float, sigma: float, q: float = 0.0):
        check_positive(s0=s0)
        check_non_negative(sigma=sigma)
        self.s0 = float(s0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.q = float(q)

    def _simulate(self, grid, normals, rng):
        dt = grid.dt
        drift = (self.mu - self.q - 0.5 * self.sigma**2) * dt
        increments = drift + self.sigma * np.sqrt(dt) * normals[:, :, 0]
        log_s = np.concatenate([np.zeros((normals.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1)
        return {"spot": self.s0 * np.exp(log_s)}

    def mean(self, t):
        """Expected value ``E[S_t]``."""
        return self.s0 * np.exp((self.mu - self.q) * np.asarray(t, dtype=float))

    def variance(self, t):
        """Variance ``Var[S_t]``."""
        t = np.asarray(t, dtype=float)
        return self.s0**2 * np.exp(2 * (self.mu - self.q) * t) * np.expm1(self.sigma**2 * t)
