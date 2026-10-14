"""Base class for stochastic processes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .grid import TimeGrid
from .random import make_rng, standard_normals
from .result import SimulationResult


class StochasticProcess(ABC):
    """A process that can be simulated on a time grid.

    Subclasses set ``state_names`` and ``n_factors`` and implement
    :meth:`_simulate`, which receives standard normal draws of shape
    ``(n_paths, n_steps, n_factors)`` and returns one array per state variable
    of shape ``(n_paths, n_steps + 1)``.
    """

    state_names: tuple = ("value",)
    n_factors: int = 1

    def simulate(
        self,
        T: float = 1.0,
        n_steps: int = 252,
        n_paths: int = 10_000,
        *,
        seed=None,
        antithetic: bool = False,
        grid: TimeGrid | None = None,
        normals: np.ndarray | None = None,
    ) -> SimulationResult:
        """Simulate paths.

        Parameters
        ----------
        T, n_steps : float, int
            Horizon in years and number of equal steps (ignored if ``grid`` is given).
        n_paths : int
            Number of paths (ignored if ``normals`` is given).
        seed : int or numpy.random.Generator, optional
            Seed for reproducible results.
        antithetic : bool
            Use antithetic variates (``n_paths`` must be even).
        grid : TimeGrid, optional
            Custom, possibly non-uniform, time grid.
        normals : ndarray, optional
            Pre-generated standard normals ``(n_paths, n_steps, n_factors)``,
            e.g. to drive several processes with common random numbers.
        """
        grid = grid if grid is not None else TimeGrid.uniform(T, n_steps)
        rng = make_rng(seed)
        if normals is None:
            normals = standard_normals(rng, n_paths, grid.n_steps, self.n_factors,
                                       antithetic=antithetic)
        else:
            normals = np.asarray(normals, dtype=float)
            if normals.ndim == 2:
                normals = normals[:, :, None]
            if normals.shape[1:] != (grid.n_steps, self.n_factors):
                raise ValueError(
                    f"normals must have shape (n_paths, {grid.n_steps}, {self.n_factors})")
        paths = self._simulate(grid, normals, rng)
        return SimulationResult(grid.times.copy(), paths, info={"process": repr(self)})

    @abstractmethod
    def _simulate(self, grid: TimeGrid, normals: np.ndarray, rng: np.random.Generator) -> dict:
        """Return a dict of paths, one ``(n_paths, n_steps + 1)`` array per state."""

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"


def check_positive(**kwargs) -> None:
    """Raise ``ValueError`` unless every keyword value is > 0."""
    for name, value in kwargs.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value}")


def check_non_negative(**kwargs) -> None:
    """Raise ``ValueError`` unless every keyword value is >= 0."""
    for name, value in kwargs.items():
        if not value >= 0:
            raise ValueError(f"{name} must be non-negative, got {value}")


def check_correlation(rho: float, name: str = "rho") -> None:
    """Raise ``ValueError`` unless ``-1 <= rho <= 1``."""
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"{name} must lie in [-1, 1], got {rho}")
