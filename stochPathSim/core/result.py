"""Container for simulated paths."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    """Simulated paths of one or more state variables on a common time grid.

    Each entry of ``paths`` has shape ``(n_paths, n_times)``; column 0 holds the
    initial state at ``t = 0``.
    """

    times: np.ndarray
    paths: dict
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        for name, p in self.paths.items():
            if p.ndim != 2 or p.shape[1] != self.times.size:
                raise ValueError(f"paths[{name!r}] must have shape (n_paths, {self.times.size})")

    @property
    def names(self) -> tuple:
        return tuple(self.paths)

    @property
    def primary(self) -> str:
        """Name of the main state variable (the first one)."""
        return self.names[0]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.paths[name]

    def get(self, name: str | None = None) -> np.ndarray:
        return self.paths[name or self.primary]

    @property
    def n_paths(self) -> int:
        return self.get().shape[0]

    @property
    def n_steps(self) -> int:
        return self.times.size - 1

    def terminal(self, name: str | None = None) -> np.ndarray:
        """Values at the last grid time."""
        return self.get(name)[:, -1]

    def at(self, t: float, name: str | None = None) -> np.ndarray:
        """Values at grid time ``t``."""
        i = int(np.argmin(np.abs(self.times - t)))
        if abs(self.times[i] - t) > 1e-12:
            raise ValueError(f"{t} is not a grid time")
        return self.get(name)[:, i]

    def mean(self, name: str | None = None) -> np.ndarray:
        """Cross-sectional mean at each time."""
        return self.get(name).mean(axis=0)

    def std(self, name: str | None = None) -> np.ndarray:
        """Cross-sectional sample standard deviation at each time."""
        return self.get(name).std(axis=0, ddof=1)

    def quantile(self, q, name: str | None = None) -> np.ndarray:
        """Cross-sectional quantile(s) at each time, ``q`` in [0, 1]."""
        return np.quantile(self.get(name), q, axis=0)

    def to_frame(self, name: str | None = None, max_paths: int | None = None):
        """Paths as a ``pandas.DataFrame`` indexed by time, one column per path."""
        import pandas as pd

        p = self.get(name)
        if max_paths is not None:
            p = p[:max_paths]
        return pd.DataFrame(p.T, index=pd.Index(self.times, name="time"),
                            columns=[f"path_{i}" for i in range(p.shape[0])])

    def plot(self, name: str | None = None, n_paths: int = 20, ax=None, **kwargs):
        """Plot a sample of paths with the mean and the 1%-99% quantile band."""
        from .. import plotting

        return plotting.plot_paths(self, name=name, n_paths=n_paths, ax=ax, **kwargs)

    def __repr__(self) -> str:
        return (f"SimulationResult(names={self.names}, n_paths={self.n_paths}, "
                f"n_steps={self.n_steps}, T={self.times[-1]:g})")
