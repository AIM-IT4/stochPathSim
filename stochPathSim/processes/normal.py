"""Arithmetic Brownian motion (the normal/Bachelier model)."""

from __future__ import annotations

import numpy as np

from ..core.process import StochasticProcess, check_non_negative


class ArithmeticBrownianMotion(StochasticProcess):
    r"""Exact simulation of ``dX = mu dt + sigma dW``.

    Values may be negative. ``mu`` is an absolute drift per year and ``sigma``
    is an absolute volatility per square-root year, both in the units of X.
    For Bachelier pricing of a forward under its forward measure, use ``mu=0``
    and discount the payoff separately. Setting ``mu`` to the risk-free rate
    does *not* produce a risk-neutral spot-price process.

    The state is named ``spot`` for compatibility with the existing payoffs.
    """

    state_names = ("spot",)
    n_factors = 1

    def __init__(self, x0: float, mu: float, sigma: float):
        if not np.all(np.isfinite([x0, mu, sigma])):
            raise ValueError("x0, mu and sigma must be finite")
        check_non_negative(sigma=sigma)
        self.x0 = float(x0)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def _simulate(self, grid, normals, rng):
        increments = self.mu * grid.dt + self.sigma * np.sqrt(grid.dt) * normals[:, :, 0]
        changes = np.concatenate(
            [np.zeros((normals.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1)
        return {"spot": self.x0 + changes}

    def mean(self, t):
        """Expected value ``x0 + mu*t``."""
        return self.x0 + self.mu * np.asarray(t, dtype=float)

    def variance(self, t):
        """Variance ``sigma**2 * t``."""
        return self.sigma**2 * np.asarray(t, dtype=float)
