"""Discrete-observation GARCH and GJR-GARCH volatility-clustering models."""

from __future__ import annotations

import numpy as np

from ..core.process import StochasticProcess


class GARCH11(StochasticProcess):
    r"""GARCH(1,1) conditional volatility with lognormal price updates.

    At observation k, the annualised conditional variance rate is v[k].
    With a standard-normal innovation z[k]:

        log(S[k+1]/S[k]) = (mu-q-v[k]/2)*dt[k] + sqrt(v[k]*dt[k])*z[k]
        v[k+1] = omega + (alpha*z[k]**2 + beta)*v[k]

    Parameters omega, alpha and beta are per observation, not per year:
    changing the number of observation steps changes the variance dynamics.
    Prices are simulated with exact conditional lognormal transitions, while
    the variance evolves by a discrete GARCH recursion. This is not a
    continuous-time diffusion and cannot be grid-refined without changing
    its interpretation. Non-uniform time grids are permitted as long as
    each interval represents one observation.

    omega has units of annualised variance per observation; v0 is the
    annualised variance rate at t=0. Stationarity alpha+beta<1 is required.
    This is a statistical simulation, NOT automatically a risk-neutral
    option-pricing model; no variance risk premium is specified.
    """

    state_names = ("spot", "variance")
    n_factors = 1

    def __init__(self, s0: float, mu: float, omega: float, alpha: float, beta: float,
                 v0: float | None = None, q: float = 0.0):
        values = [s0, mu, omega, alpha, beta, q]
        if v0 is not None:
            values.append(v0)
        if not np.all(np.isfinite(values)):
            raise ValueError("all GARCH parameters must be finite")
        if s0 <= 0 or omega <= 0 or alpha < 0 or beta < 0:
            raise ValueError("s0, omega must be positive and alpha, beta non-negative")
        if alpha + beta >= 1:
            raise ValueError("GARCH parameters must satisfy alpha + beta < 1")
        if v0 is not None and v0 <= 0:
            raise ValueError("v0 must be positive")
        self.s0 = float(s0)
        self.mu = float(mu)
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.q = float(q)
        self.v0 = float(v0) if v0 is not None else self.long_run_variance

    @property
    def long_run_variance(self) -> float:
        """Unconditional stationary annualised variance rate."""
        return self.omega / (1.0 - self.alpha - self.beta)

    def _simulate(self, grid, normals, rng):
        n_paths, n_steps, _ = normals.shape
        spot = np.empty((n_paths, n_steps + 1))
        variance = np.empty_like(spot)
        spot[:, 0], variance[:, 0] = self.s0, self.v0
        for k, dt in enumerate(grid.dt):
            v = variance[:, k]
            z = normals[:, k, 0]
            spot[:, k + 1] = spot[:, k] * np.exp(
                (self.mu - self.q - 0.5*v)*dt + np.sqrt(v*dt)*z)
            variance[:, k + 1] = self.omega + (self.alpha*z**2 + self.beta)*v
        return {"spot": spot, "variance": variance}

    def expected_variance(self, n: int):
        """Expected variance rate at observation n, conditional on v0."""
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a non-negative integer")
        phi = self.alpha + self.beta
        return self.long_run_variance + (self.v0 - self.long_run_variance)*phi**int(n)


class GJRGARCH11(GARCH11):
    r"""GJR-GARCH(1,1) with asymmetric response to negative shocks.

        v[k+1] = omega + (alpha + gamma*1[z[k]<0])*v[k]*z[k]**2 + beta*v[k]

    Under symmetric standard-normal shocks the persistence coefficient is
    alpha + beta + gamma/2, which must be strictly below one. A positive
    gamma makes negative innovations cause more subsequent volatility.
    Like GARCH11 this is a per-observation discrete-time statistical model.
    """

    def __init__(self, s0: float, mu: float, omega: float, alpha: float, beta: float,
                 gamma: float, v0: float | None = None, q: float = 0.0):
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative")
        if alpha + beta + gamma/2 >= 1:
            raise ValueError("GJR persistence alpha + beta + gamma/2 must be < 1")
        self.gamma = float(gamma)
        super().__init__(s0, mu, omega, alpha, beta, v0=v0, q=q)

    @property
    def long_run_variance(self) -> float:
        """Stationary expected variance under symmetric innovations."""
        return self.omega / (1.0 - self.alpha - self.beta - self.gamma/2)

    def _simulate(self, grid, normals, rng):
        n_paths, n_steps, _ = normals.shape
        spot = np.empty((n_paths, n_steps + 1))
        variance = np.empty_like(spot)
        spot[:, 0], variance[:, 0] = self.s0, self.v0
        for k, dt in enumerate(grid.dt):
            v = variance[:, k]
            z = normals[:, k, 0]
            spot[:, k + 1] = spot[:, k] * np.exp(
                (self.mu - self.q - 0.5*v)*dt + np.sqrt(v*dt)*z)
            variance[:, k + 1] = self.omega + (
                self.alpha*z**2 + self.gamma*np.where(z < 0, z**2, 0.0) + self.beta)*v
        return {"spot": spot, "variance": variance}

    def expected_variance(self, n: int):
        """Expected variance rate n observations ahead."""
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a non-negative integer")
        phi = self.alpha + self.beta + self.gamma/2
        return self.long_run_variance + (self.v0 - self.long_run_variance)*phi**int(n)
