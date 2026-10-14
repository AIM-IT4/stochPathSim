"""Default intensity (hazard rate) models and default-time simulation."""

from __future__ import annotations

import numpy as np

from ..core.process import StochasticProcess, check_non_negative, check_positive
from ..core.random import make_rng


class _IntensityModel(StochasticProcess):
    """Common default-time simulation for intensity models with states ``intensity`` and ``survival``."""

    state_names = ("intensity", "survival")

    @staticmethod
    def default_times(result, seed=None) -> np.ndarray:
        r"""Default times :math:`\tau = \inf\{t : \int_0^t \lambda_s ds \ge E\}`, ``E ~ Exp(1)``.

        The cumulative hazard is interpolated linearly between grid times. Paths that do
        not default before the last grid time get ``np.inf``.
        """
        rng = make_rng(seed)
        hazard = -np.log(result["survival"])
        times = result.times
        e = rng.exponential(1.0, size=hazard.shape[0])
        hit = hazard >= e[:, None]
        tau = np.full(hazard.shape[0], np.inf)
        rows = np.flatnonzero(hit.any(axis=1))
        k = hit[rows].argmax(axis=1)
        h0 = hazard[rows, k - 1]
        h1 = hazard[rows, k]
        w = (e[rows] - h0) / np.where(h1 > h0, h1 - h0, 1.0)
        tau[rows] = times[k - 1] + w * (times[k] - times[k - 1])
        return tau


class ConstantIntensity(_IntensityModel):
    r"""Constant default intensity ``lam``: survival probability :math:`e^{-\lambda t}`."""

    n_factors = 1

    def __init__(self, lam: float):
        check_non_negative(lam=lam)
        self.lam = float(lam)

    def _simulate(self, grid, normals, rng):
        n_paths = normals.shape[0]
        t = np.broadcast_to(grid.times, (n_paths, grid.times.size))
        return {"intensity": np.full(t.shape, self.lam), "survival": np.exp(-self.lam * t)}

    def survival_probability(self, t):
        return np.exp(-self.lam * np.asarray(t, dtype=float))


class CIRIntensity(_IntensityModel):
    r"""Stochastic default intensity following a CIR process

    .. math:: d\lambda_t = \kappa(\theta - \lambda_t)\,dt + \sigma\sqrt{\lambda_t}\,dW_t

    The intensity is simulated exactly (non-central chi-square transitions, drawn
    from the generator); the pathwise survival ``exp(-∫λ dt)`` uses the trapezoid
    rule on the grid. Its expectation is the CIR bond price, see
    :meth:`survival_probability`.
    """

    n_factors = 1

    def __init__(self, lam0: float, kappa: float, theta: float, sigma: float):
        check_non_negative(lam0=lam0)
        check_positive(kappa=kappa, theta=theta, sigma=sigma)
        self.lam0 = float(lam0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)

    def _simulate(self, grid, normals, rng):
        n_paths, n_steps, _ = normals.shape
        dt = grid.dt
        lam = np.empty((n_paths, n_steps + 1))
        lam[:, 0] = self.lam0
        df = 4 * self.kappa * self.theta / self.sigma**2
        for k in range(n_steps):
            c = self.sigma**2 * -np.expm1(-self.kappa * dt[k]) / (4 * self.kappa)
            nonc = lam[:, k] * np.exp(-self.kappa * dt[k]) / c
            lam[:, k + 1] = c * rng.noncentral_chisquare(df, nonc)
        integral = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(0.5 * (lam[:, 1:] + lam[:, :-1]) * dt, axis=1)], axis=1)
        return {"intensity": lam, "survival": np.exp(-integral)}

    def survival_probability(self, t):
        """Risk-neutral survival probability ``E[exp(-∫λ)]`` (the CIR bond formula)."""
        from ..analytics.rates import cir_zcb

        return cir_zcb(self.lam0, 0.0, t, self.kappa, self.theta, self.sigma)
