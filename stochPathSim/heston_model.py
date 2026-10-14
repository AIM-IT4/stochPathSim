"""Version 1 interface, kept for backward compatibility.

Prefer :class:`stochPathSim.Heston`, which also takes a drift and a dividend yield.
"""

from __future__ import annotations

from ._legacy import legacy_grid
from .processes.equity import Heston


class HestonModel:
    """Heston model with zero drift, ``n`` time points from 0 to ``T``."""

    def __init__(self, v0, kappa, theta, sigma, rho, s0, T, n, paths, seed=None):
        self.v0, self.kappa, self.theta, self.sigma, self.rho = v0, kappa, theta, sigma, rho
        self.s0, self.T, self.n, self.paths, self.seed = s0, T, n, paths, seed

    def simulate(self):
        """Return ``(spot, variance)``, each of shape ``(paths, n)``."""
        model = Heston(self.s0, 0.0, self.v0, self.kappa, self.theta, self.sigma, self.rho)
        res = model.simulate(grid=legacy_grid(self.T, self.n), n_paths=self.paths, seed=self.seed)
        return res["spot"], res["variance"]
