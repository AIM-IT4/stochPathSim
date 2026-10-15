"""Exact finite-grid fractional Brownian motion via fractional-Gaussian-noise covariance."""

from __future__ import annotations

import numpy as np

from ..core.process import StochasticProcess


class FractionalBrownianMotion(StochasticProcess):
    r"""Fractional Brownian motion with Hurst exponent 0 < H < 1.

    X(t) = x0 + drift*t + sigma*B_H(t), with covariance

        Cov(B_H(t), B_H(s)) = (t**(2H)+s**(2H)-abs(t-s)**(2H))/2.

    We Cholesky-factor the increment covariance on the requested grid,
    then cumulatively sum the correlated increments. This samples the
    exact finite-dimensional Gaussian law, including on non-uniform grids,
    subject to floating-point factorisation accuracy.

    For H = 0.5 the increments reduce exactly to independent Brownian
    increments and supplied normals match standard Brownian simulation.
    For H != 0.5 the process is NOT a semimartingale; the resulting
    paths are intended for long-memory/roughness studies, NOT directly
    as an arbitrage-free traded-asset or derivative-pricing model.

    The Cholesky decomposition requires O(n_steps**3) preprocessing and
    O(n_steps**2) memory; use modest grids (a few hundred observations).
    """

    state_names = ("value",)
    n_factors = 1

    def __init__(self, x0: float = 0.0, hurst: float = 0.5,
                 sigma: float = 1.0, drift: float = 0.0):
        if not np.all(np.isfinite([x0, hurst, sigma, drift])):
            raise ValueError("all fBM parameters must be finite")
        if not 0 < hurst < 1:
            raise ValueError("hurst must lie in (0, 1)")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.x0 = float(x0)
        self.hurst = float(hurst)
        self.sigma = float(sigma)
        self.drift = float(drift)

    def increment_covariance(self, grid):
        """Covariance matrix of unit-scale increments over a TimeGrid."""
        left = grid.times[:-1]
        right = grid.times[1:]
        h2 = 2.0 * self.hurst
        cov = 0.5 * (
            np.abs(right[:, None] - left[None, :])**h2
            + np.abs(left[:, None] - right[None, :])**h2
            - np.abs(right[:, None] - right[None, :])**h2
            - np.abs(left[:, None] - left[None, :])**h2
        )
        return 0.5 * (cov + cov.T)

    def _simulate(self, grid, normals, rng):
        n_paths = normals.shape[0]
        if self.sigma == 0:
            return {"value": np.broadcast_to(
                self.x0 + self.drift * grid.times, (n_paths, grid.n_steps + 1)).copy()}
        if self.hurst == 0.5:
            increments = np.sqrt(grid.dt) * normals[:, :, 0]
        else:
            cov = self.increment_covariance(grid)
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as exc:
                raise ValueError("increment covariance could not be factorised") from exc
            increments = normals[:, :, 0] @ chol.T
        cumsum = np.concatenate(
            (np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)), axis=1)
        return {"value": self.x0 + self.drift*grid.times + self.sigma*cumsum}

    def mean(self, t):
        """Exact expected value at scalar or array times."""
        return self.x0 + self.drift * np.asarray(t, dtype=float)

    def variance(self, t):
        """Exact variance at scalar or array times."""
        t = np.asarray(t, dtype=float)
        if np.any(t < 0) or np.any(~np.isfinite(t)):
            raise ValueError("times must be finite and non-negative")
        return self.sigma**2 * t**(2*self.hurst)
