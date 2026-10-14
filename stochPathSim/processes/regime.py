"""Regime-switching geometric Brownian motion."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..core.process import StochasticProcess, check_positive


class RegimeSwitchingGBM(StochasticProcess):
    r"""GBM whose drift and volatility follow a continuous-time Markov chain

    .. math:: \frac{dS_t}{S_t} = \mu_{R_t}\,dt + \sigma_{R_t}\,dW_t

    where :math:`R_t` jumps between ``n`` regimes with generator matrix ``Q``
    (``Q[i, j]`` is the rate of switching from ``i`` to ``j``, rows sum to zero).

    The regime chain is simulated exactly on the grid with the transition
    matrices ``expm(Q dt)``. The spot uses the regime at the start of each step,
    so it is exact when regimes only change on grid dates and has an ``O(dt)``
    error otherwise; use steps that are short compared with the mean regime
    duration ``1 / -Q[i, i]``.

    Parameters
    ----------
    s0 : float
        Initial spot.
    mu, sigma : array_like
        Drift and volatility in each regime.
    generator : array_like
        ``n x n`` generator matrix ``Q``.
    initial_regime : int
        Regime at ``t = 0``.
    """

    state_names = ("spot", "regime")
    n_factors = 1

    def __init__(self, s0: float, mu, sigma, generator, initial_regime: int = 0):
        check_positive(s0=s0)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        q = np.asarray(generator, dtype=float)
        n = mu.size
        if mu.ndim != 1 or sigma.shape != mu.shape or q.shape != (n, n):
            raise ValueError("mu, sigma must have length n and generator must be n x n")
        if np.any(sigma < 0.0):
            raise ValueError("sigma must be non-negative")
        off = q - np.diag(np.diag(q))
        if np.any(off < 0.0) or not np.allclose(q.sum(axis=1), 0.0):
            raise ValueError("generator needs non-negative off-diagonal rates and rows summing to zero")
        if not 0 <= int(initial_regime) < n:
            raise ValueError(f"initial_regime must lie in [0, {n - 1}]")
        self.s0 = float(s0)
        self.mu = mu
        self.sigma = sigma
        self.generator = q
        self.initial_regime = int(initial_regime)

    @property
    def n_regimes(self) -> int:
        return self.mu.size

    def transition_matrix(self, t: float) -> np.ndarray:
        """Regime transition probabilities over a horizon ``t``: ``expm(Q t)``."""
        return expm(self.generator * t)

    def _simulate(self, grid, normals, rng):
        n_paths, n_steps, _ = normals.shape
        dt = grid.dt
        regime = np.empty((n_paths, n_steps + 1), dtype=int)
        regime[:, 0] = self.initial_regime
        cache: dict = {}
        uniforms = rng.random((n_paths, n_steps))
        for k in range(n_steps):
            key = round(float(dt[k]), 14)
            if key not in cache:
                cum = np.cumsum(self.transition_matrix(dt[k]), axis=1)
                cum[:, -1] = 1.0
                cache[key] = cum
            cum_rows = cache[key][regime[:, k]]
            regime[:, k + 1] = (uniforms[:, k, None] > cum_rows).sum(axis=1)

        r = regime[:, :-1]
        mu, sig = self.mu[r], self.sigma[r]
        inc = (mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * normals[:, :, 0]
        log_s = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)
        return {"spot": self.s0 * np.exp(log_s), "regime": regime.astype(float)}

    def mean(self, t):
        """Expected spot ``E[S_t] = s0 e_i^T expm((Q + diag(mu)) t) 1`` (Feynman-Kac)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        a = self.generator + np.diag(self.mu)
        out = np.array([expm(a * ti)[self.initial_regime].sum() for ti in t]) * self.s0
        return out if out.size > 1 else float(out[0])
