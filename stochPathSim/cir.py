"""Version 1 interface, kept for backward compatibility.

Prefer :class:`stochPathSim.CIR`, which also returns pathwise discount factors.
"""

from __future__ import annotations

from ._legacy import legacy_grid
from .processes.rates import CIR


class CoxIngersollRossModel:
    """CIR short rate with ``n`` time points from 0 to ``T``."""

    def __init__(self, r0, kappa, theta, sigma, T, n, paths, seed=None):
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.T, self.n, self.paths, self.seed = T, n, paths, seed

    def simulate(self):
        """Return short-rate paths of shape ``(paths, n)``."""
        res = CIR(self.r0, self.kappa, self.theta, self.sigma).simulate(
            grid=legacy_grid(self.T, self.n), n_paths=self.paths, seed=self.seed)
        return res["rate"]
