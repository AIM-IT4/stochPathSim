"""Version 1 interface, kept for backward compatibility.

Prefer :class:`stochPathSim.BlackDermanToy`.

Version 1 simulated an arithmetic random walk. The rate is now lognormal, as in the
Black-Derman-Toy model.
"""

from __future__ import annotations

from ._legacy import legacy_grid
from .processes.rates import BlackDermanToy


class BlackDermanToyModel:
    """Lognormal (Black-Derman-Toy) short rate with zero drift in ``ln r``."""

    def __init__(self, r0, sigma, T, n, paths, seed=None):
        self.r0, self.sigma = r0, sigma
        self.T, self.n, self.paths, self.seed = T, n, paths, seed

    def simulate(self):
        """Return short-rate paths of shape ``(paths, n)``."""
        res = BlackDermanToy(self.r0, self.sigma).simulate(grid=legacy_grid(self.T, self.n),
                                                           n_paths=self.paths, seed=self.seed)
        return res["rate"]
