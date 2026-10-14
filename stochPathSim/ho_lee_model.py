"""Version 1 interface, kept for backward compatibility.

Prefer :class:`stochPathSim.HoLee`, which fits any initial zero curve.
"""

from __future__ import annotations

import numpy as np

from ._legacy import legacy_grid
from .core import make_rng


class HoLeeModel:
    """Ho-Lee short rate ``dr = theta dt + sigma dW`` with constant ``theta``."""

    def __init__(self, r0, theta, sigma, T, n, paths, seed=None):
        self.r0, self.theta, self.sigma = r0, theta, sigma
        self.T, self.n, self.paths, self.seed = T, n, paths, seed

    def simulate(self):
        """Return short-rate paths of shape ``(paths, n)`` (exact simulation)."""
        grid = legacy_grid(self.T, self.n)
        z = make_rng(self.seed).standard_normal((self.paths, grid.n_steps))
        steps = self.theta * grid.dt + self.sigma * np.sqrt(grid.dt) * z
        return self.r0 + np.concatenate([np.zeros((self.paths, 1)), np.cumsum(steps, axis=1)], axis=1)
