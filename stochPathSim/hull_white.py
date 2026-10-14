"""Version 1 interface, kept for backward compatibility.

Prefer :class:`stochPathSim.HullWhite`, which fits any initial zero curve.

Version 1 simulated an Ornstein-Uhlenbeck process reverting to zero. The model is now a
proper Hull-White model fitted to a flat curve at ``r0``.
"""

from __future__ import annotations

from ._legacy import legacy_grid
from .curves import ZeroCurve
from .processes.rates import HullWhite


class HullWhiteModel:
    """Hull-White short rate fitted to a flat curve at ``r0``, ``n`` time points from 0 to ``T``."""

    def __init__(self, r0, a, sigma, T, n, paths, seed=None):
        self.r0, self.a, self.sigma = r0, a, sigma
        self.T, self.n, self.paths, self.seed = T, n, paths, seed

    def simulate(self):
        """Return short-rate paths of shape ``(paths, n)``."""
        model = HullWhite(self.a, self.sigma, ZeroCurve.flat(self.r0))
        res = model.simulate(grid=legacy_grid(self.T, self.n), n_paths=self.paths, seed=self.seed)
        return res["rate"]
