"""Kou double-exponential jump diffusion."""

from __future__ import annotations

import numpy as np

from ..core.process import StochasticProcess, check_non_negative, check_positive


def kou_jump_compensator(p_up: float, eta_up: float, eta_down: float) -> float:
    """``E[e^Y] - 1`` for a double-exponential log-jump ``Y``."""
    return p_up * eta_up / (eta_up - 1.0) + (1.0 - p_up) * eta_down / (eta_down + 1.0) - 1.0


class KouJumpDiffusion(StochasticProcess):
    r"""Kou (2002) double-exponential jump diffusion

    .. math:: \frac{dS_t}{S_{t^-}} = (\mu - q - \lambda\zeta)\,dt + \sigma\,dW_t + (e^{Y} - 1)\,dN_t

    with Poisson jumps of intensity ``lam``. A log-jump ``Y`` is ``+Exp(eta_up)``
    with probability ``p_up`` and ``-Exp(eta_down)`` otherwise, so jumps are
    asymmetric and heavy-tailed (unlike Merton's normal jumps).
    ``zeta = E[e^Y] - 1`` compensates the jumps so that ``E[S_t] = s0 exp((mu - q) t)``.
    Simulated exactly: per step the up and down jump counts are binomial splits of a
    Poisson count, and sums of exponentials are gamma distributed.

    Parameters
    ----------
    eta_up : float
        Rate of upward jumps; must exceed 1 for ``E[S_t]`` to be finite.
    eta_down : float
        Rate of downward jumps (mean down-jump size ``1/eta_down``).
    """

    state_names = ("spot",)
    n_factors = 1

    def __init__(self, s0: float, mu: float, sigma: float, lam: float, p_up: float,
                 eta_up: float, eta_down: float, q: float = 0.0):
        check_positive(s0=s0, eta_down=eta_down)
        check_non_negative(sigma=sigma, lam=lam)
        if not 0.0 <= p_up <= 1.0:
            raise ValueError(f"p_up must lie in [0, 1], got {p_up}")
        if not eta_up > 1.0:
            raise ValueError(f"eta_up must exceed 1, got {eta_up}")
        self.s0 = float(s0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.p_up = float(p_up)
        self.eta_up = float(eta_up)
        self.eta_down = float(eta_down)
        self.q = float(q)

    @property
    def compensator(self) -> float:
        return kou_jump_compensator(self.p_up, self.eta_up, self.eta_down)

    def _simulate(self, grid, normals, rng):
        n_paths, n_steps, _ = normals.shape
        dt = grid.dt
        drift = (self.mu - self.q - self.lam * self.compensator - 0.5 * self.sigma**2) * dt
        inc = drift + self.sigma * np.sqrt(dt) * normals[:, :, 0]
        counts = rng.poisson(self.lam * dt, size=(n_paths, n_steps))
        n_up = rng.binomial(counts, self.p_up)
        n_down = counts - n_up
        up = np.where(n_up > 0, rng.gamma(np.maximum(n_up, 1), 1.0 / self.eta_up), 0.0)
        down = np.where(n_down > 0, rng.gamma(np.maximum(n_down, 1), 1.0 / self.eta_down), 0.0)
        inc = inc + up - down
        log_s = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)
        return {"spot": self.s0 * np.exp(log_s)}

    def mean(self, t):
        """Expected value ``E[S_t]``."""
        return self.s0 * np.exp((self.mu - self.q) * np.asarray(t, dtype=float))
