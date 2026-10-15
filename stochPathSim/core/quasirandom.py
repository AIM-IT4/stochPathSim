"""Sobol quasi-Monte-Carlo normals, with optional Brownian-bridge ordering."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .grid import TimeGrid
from .random import cholesky


def _brownian_bridge(z: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Map Sobol-normal dimensions into independent normalized Brownian increments.

    Dimension zero drives the terminal Brownian motion; subsequent dimensions
    draw progressively finer Brownian-bridge midpoints. The underlying paths
    retain the exact Wiener Gaussian covariance for normal inputs.
    """
    n_paths, n_steps, n_factors = z.shape
    w = np.zeros((n_paths, n_steps + 1, n_factors))
    w[:, -1, :] = np.sqrt(times[-1]) * z[:, 0, :]
    queue = deque([(0, n_steps)])
    index = 1
    while queue:
        a, b = queue.popleft()
        if b - a <= 1:
            continue
        middle = (a + b)//2
        t0, tm, t1 = times[a], times[middle], times[b]
        weight = (tm - t0)/(t1 - t0)
        sd = np.sqrt((tm - t0)*(t1 - tm)/(t1 - t0))
        w[:, middle, :] = (1 - weight)*w[:, a, :] + weight*w[:, b, :] + sd*z[:, index, :]
        index += 1
        queue.append((a, middle))
        queue.append((middle, b))
    assert index == n_steps
    return np.diff(w, axis=1) / np.sqrt(np.diff(times))[None, :, None]


def sobol_normals(n_paths: int, n_steps: int, n_factors: int = 1, *,
                  seed: int | None = None, scramble: bool = True,
                  antithetic: bool = False, brownian_bridge: bool = False,
                  grid: TimeGrid | None = None, correlation=None) -> np.ndarray:
    """Generate QMC normals accepted by StochasticProcess.simulate(normals=...).

    n_paths must be a power of two (Sobol balance property). Scrambling
    is enabled by default; use independent seeds for confidence intervals
    across multiple QMC scrambles, NOT the pathwise sample standard error
    of one scrambled Sobol point set. With antithetic=True, half the
    samples are generated and their negatives are appended.

    brownian_bridge=True maps early Sobol dimensions to long-horizon
    Brownian motions, reducing effective dimension for many path-dependent
    payoffs. grid selects the actual non-uniform Brownian bridge times;
    otherwise a unit-horizon uniform grid is used. With the bridge disabled,
    grid is only shape-validated and has no effect on the normals.

    correlation applies Cholesky factor correlation to the output factors,
    after Brownian-bridge construction.
    """
    for name, value in (("n_paths", n_paths), ("n_steps", n_steps), ("n_factors", n_factors)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if n_paths & (n_paths - 1):
        raise ValueError("Sobol n_paths must be a power of two")
    if antithetic and n_paths < 2:
        raise ValueError("antithetic sampling needs at least two paths")
    if grid is not None and (not isinstance(grid, TimeGrid) or grid.n_steps != n_steps):
        raise ValueError("grid must be a TimeGrid with n_steps intervals")
    n_base = n_paths // 2 if antithetic else n_paths
    sampler = qmc.Sobol(d=n_steps*n_factors, scramble=scramble, seed=seed)
    uniform = sampler.random_base2(m=n_base.bit_length() - 1)
    # Avoid infinities at the endpoints of the inverse normal CDF.
    eps = np.finfo(float).eps
    z = ndtri(np.clip(uniform, eps, 1 - eps)).reshape(n_base, n_steps, n_factors)
    if brownian_bridge:
        times = grid.times if grid is not None else np.linspace(0, 1, n_steps + 1)
        z = _brownian_bridge(z, times)
    if antithetic:
        z = np.concatenate((z, -z), axis=0)
    if correlation is not None:
        z = z @ cholesky(correlation, n_factors).T
    return z
