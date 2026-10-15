"""Displaced (shifted-lognormal) diffusion for forwards that may be negative."""

from __future__ import annotations

import numpy as np

from ..core.process import StochasticProcess


class DisplacedDiffusion(StochasticProcess):
    r"""Exact displaced diffusion of a forward or other shifted underlying.

    The positive shifted variable X = S + shift follows

        dX / X = (mu - q) dt + sigma dW.

    Consequently S is bounded below by -shift but may be negative, and

        E[S_t] = (s0 + shift) exp((mu - q) t) - shift.

    For an *undiscounted forward* under its forward measure, set mu=q=0;
    the shifted process then is a martingale. mu is the drift of X, NOT
    the conventional proportional drift of S when shift is nonzero.
    Price a European option using displaced_diffusion_price(F, ...).

    Parameters
    ----------
    s0 : float
        Initial forward/underlying, possibly negative.
    mu, q : float
        Annual proportional drift and carry of the shifted variable.
    sigma : float
        Annual lognormal volatility of the shifted variable.
    shift : float
        Displacement such that s0 + shift > 0.
    """

    state_names = ("spot",)
    n_factors = 1

    def __init__(self, s0: float, mu: float, sigma: float, shift: float, q: float = 0.0):
        if not np.all(np.isfinite([s0, mu, sigma, shift, q])):
            raise ValueError("all parameters must be finite")
        if s0 + shift <= 0.0:
            raise ValueError("s0 + shift must be positive")
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        self.s0 = float(s0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.shift = float(shift)
        self.q = float(q)

    def _simulate(self, grid, normals, rng):
        dt = grid.dt
        increments = ((self.mu - self.q - 0.5 * self.sigma**2) * dt
                      + self.sigma * np.sqrt(dt) * normals[:, :, 0])
        log_multiplier = np.concatenate(
            [np.zeros((normals.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1)
        return {"spot": (self.s0 + self.shift) * np.exp(log_multiplier) - self.shift}

    def mean(self, t):
        """Exact E[S_t] for scalar or array times."""
        return (self.s0 + self.shift) * np.exp((self.mu - self.q) * np.asarray(t)) - self.shift

    def variance(self, t):
        """Exact Var[S_t] for scalar or array times."""
        t = np.asarray(t)
        return ((self.s0 + self.shift)**2 * np.exp(2 * (self.mu - self.q) * t)
                * np.expm1(self.sigma**2 * t))
