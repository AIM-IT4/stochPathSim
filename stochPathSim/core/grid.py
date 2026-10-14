"""Simulation time grids."""

from __future__ import annotations

import numpy as np


class TimeGrid:
    """An increasing set of simulation times starting at 0.

    Parameters
    ----------
    times : array_like
        Strictly increasing times in years, starting at 0.
    """

    def __init__(self, times):
        times = np.array(times, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("a time grid needs at least two times")
        if times[0] != 0.0:
            raise ValueError("a time grid must start at 0")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("grid times must be strictly increasing")
        times.setflags(write=False)
        self._times = times

    @classmethod
    def uniform(cls, T: float, n_steps: int) -> TimeGrid:
        """A grid of ``n_steps`` equal steps from 0 to ``T``."""
        if not T > 0.0:
            raise ValueError("T must be positive")
        if int(n_steps) < 1:
            raise ValueError("n_steps must be at least 1")
        return cls(np.linspace(0.0, T, int(n_steps) + 1))

    @classmethod
    def with_times(cls, T: float, n_steps: int, extra_times) -> TimeGrid:
        """A uniform grid refined so that it also contains ``extra_times``."""
        base = np.linspace(0.0, T, int(n_steps) + 1)
        extra = np.asarray(extra_times, dtype=float)
        if np.any((extra < 0.0) | (extra > T)):
            raise ValueError("extra times must lie in [0, T]")
        return cls(np.unique(np.concatenate([base, extra])))

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def dt(self) -> np.ndarray:
        """Step lengths, one per step."""
        return np.diff(self._times)

    @property
    def n_steps(self) -> int:
        return self._times.size - 1

    @property
    def T(self) -> float:
        return float(self._times[-1])

    def index(self, t: float) -> int:
        """Index of the grid time equal to ``t`` (within 1e-12)."""
        i = int(np.argmin(np.abs(self._times - t)))
        if abs(self._times[i] - t) > 1e-12:
            raise ValueError(f"{t} is not a grid time")
        return i

    def __len__(self) -> int:
        return self._times.size

    def __repr__(self) -> str:
        return f"TimeGrid(T={self.T:g}, n_steps={self.n_steps})"
