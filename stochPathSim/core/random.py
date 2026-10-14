"""Random number generation for path simulation."""

from __future__ import annotations

import numpy as np


def make_rng(seed=None) -> np.random.Generator:
    """Return a NumPy ``Generator``; pass an int for reproducible results."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def standard_normals(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    n_factors: int = 1,
    *,
    antithetic: bool = False,
    correlation=None,
) -> np.ndarray:
    """Independent (or correlated) standard normal draws.

    Returns an array of shape ``(n_paths, n_steps, n_factors)``.

    Parameters
    ----------
    antithetic : bool
        Pair every draw with its negative. ``n_paths`` must then be even.
    correlation : array_like, optional
        ``n_factors x n_factors`` correlation matrix applied through its Cholesky factor.
    """
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    if antithetic:
        if n_paths % 2:
            raise ValueError("antithetic sampling needs an even number of paths")
        half = rng.standard_normal((n_paths // 2, n_steps, n_factors))
        z = np.concatenate([half, -half], axis=0)
    else:
        z = rng.standard_normal((n_paths, n_steps, n_factors))
    if correlation is not None:
        z = z @ cholesky(correlation, n_factors).T
    return z


def cholesky(correlation, n_factors: int | None = None) -> np.ndarray:
    """Lower Cholesky factor of a validated correlation matrix."""
    c = np.asarray(correlation, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("correlation must be a square matrix")
    if n_factors is not None and c.shape[0] != n_factors:
        raise ValueError(f"correlation must be {n_factors}x{n_factors}")
    if not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0):
        raise ValueError("correlation must be symmetric with a unit diagonal")
    try:
        return np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("correlation matrix is not positive definite") from exc
