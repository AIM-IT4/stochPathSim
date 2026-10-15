"""Exponential normal inverse Gaussian (NIG) Levy process."""

from __future__ import annotations

import numpy as np

from ..core.process import StochasticProcess, check_non_negative, check_positive


class NormalInverseGaussian(StochasticProcess):
    r"""An exponential NIG asset model with exact increments.

    ``S_t = s0 * exp((mu - q - kappa(1))*t + L_t)``, where
    ``kappa(z) = delta * (sqrt(alpha**2-beta**2) - sqrt(alpha**2-(beta+z)**2))``
    is the per-year cumulant of the NIG Levy process L. The compensator ensures
    ``E[S_t] = s0*exp((mu-q)*t)``; use ``mu=r`` for risk-neutral spot pricing.

    ``alpha`` controls tail decay, ``beta`` controls asymmetry, and ``delta``
    controls activity per year. Require ``alpha > abs(beta)`` and
    ``alpha > abs(beta+1)`` so the model and its first exponential moment are
    well-defined. ``delta=0`` gives deterministic growth. Negative beta gives
    negatively skewed log returns. The spot remains positive.

    Each increment is ``beta*G + sqrt(G)*Z``, with independent inverse Gaussian
    clock ``G ~ IG(delta*dt/gamma, (delta*dt)**2)`` and ``gamma=sqrt(alpha**2-beta**2)``.
    This matches SciPy's ``norminvgauss(a=alpha*delta*t, b=beta*delta*t,
    scale=delta*t)`` parameterisation of L_t.

    ``normals=`` controls only Z. Clock draws use ``seed``; reuse both for
    reproducibility. Antithetic sampling pairs Z only, not the clock draws.
    """

    state_names = ("spot",)
    n_factors = 1

    def __init__(self, s0: float, mu: float, alpha: float, beta: float,
                 delta: float, q: float = 0.0):
        if not np.all(np.isfinite([s0, mu, alpha, beta, delta, q])):
            raise ValueError("all NIG parameters must be finite")
        check_positive(s0=s0, alpha=alpha)
        check_non_negative(delta=delta)
        if alpha <= max(abs(beta), abs(beta + 1.0)):
            raise ValueError("alpha must exceed both abs(beta) and abs(beta + 1)")
        self.s0 = float(s0)
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.q = float(q)

    @property
    def _gamma(self):
        return np.sqrt((self.alpha - self.beta) * (self.alpha + self.beta))

    def _cumulant(self, z):
        # Rationalised difference of square roots avoids cancellation near z=0.
        root = np.sqrt((self.alpha - self.beta - z) * (self.alpha + self.beta + z))
        return self.delta * z * (2 * self.beta + z) / (self._gamma + root)

    @property
    def compensator(self) -> float:
        """Per-year correction kappa(1) for the exponential martingale."""
        return float(self._cumulant(1.0))

    def _simulate(self, grid, normals, rng):
        dt = grid.dt
        increments = np.broadcast_to(
            (self.mu - self.q - self.compensator) * dt, normals.shape[:2]).copy()
        if self.delta > 0:
            clock = rng.wald(self.delta * dt / self._gamma, (self.delta * dt)**2,
                             size=normals.shape[:2])
            increments += self.beta * clock + np.sqrt(clock) * normals[:, :, 0]
        log_returns = np.concatenate(
            [np.zeros((normals.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1)
        return {"spot": self.s0 * np.exp(log_returns)}

    def mean(self, t):
        """Expected spot ``s0*exp((mu-q)*t)``."""
        return self.s0 * np.exp((self.mu - self.q) * np.asarray(t, dtype=float))

    def variance(self, t):
        """Spot variance; requires ``alpha > abs(beta+2)`` when delta is positive.

        Parameters outside that interior second-moment domain raise ValueError.
        """
        t = np.asarray(t, dtype=float)
        if self.delta == 0:
            return np.zeros_like(t)
        if self.alpha <= abs(self.beta + 2.0):
            raise ValueError("spot variance requires alpha > abs(beta + 2)")
        return self.mean(t)**2 * np.expm1((self._cumulant(2.0) - 2 * self.compensator) * t)

    def log_return_mean(self, t):
        """Expected log return ``E[log(S_t/s0)]``."""
        drift = self.mu - self.q - self.compensator + self.delta * self.beta / self._gamma
        return drift * np.asarray(t, dtype=float)

    def log_return_variance(self, t):
        """Variance of log returns, finite for every accepted parameter set."""
        return self.delta * self.alpha**2 / self._gamma**3 * np.asarray(t, dtype=float)
