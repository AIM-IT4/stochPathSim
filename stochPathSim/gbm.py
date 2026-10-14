"""Version 1 interface, kept for backward compatibility.

Prefer :class:`stochPathSim.GBM`.
"""

from __future__ import annotations

from ._legacy import legacy_grid
from .processes.gbm import GBM


class GeometricBrownianMotion:
    """Geometric Brownian motion with ``n`` time points from 0 to ``T``."""

    def __init__(self, mu, sigma, s0, T, n, paths, seed=None):
        self.mu, self.sigma, self.s0 = mu, sigma, s0
        self.T, self.n, self.paths, self.seed = T, n, paths, seed

    def simulate(self):
        """Return ``(time, paths)``: ``n`` times and an array of shape ``(paths, n)``."""
        res = GBM(self.s0, self.mu, self.sigma).simulate(grid=legacy_grid(self.T, self.n),
                                                        n_paths=self.paths, seed=self.seed)
        return res.times, res.get()
