"""Monte Carlo pricing of path-dependent payoffs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..curves import ZeroCurve


@dataclass
class MCResult:
    """Monte Carlo price estimate.

    Attributes
    ----------
    price : float
        Estimated present value.
    stderr : float
        Standard error of the estimate.
    n_paths : int
        Number of simulated paths.
    samples : ndarray, optional
        The independent discounted samples the estimate averages (pair averages
        under antithetic sampling, control-variate adjusted if one was used).
    """

    price: float
    stderr: float
    n_paths: int
    samples: np.ndarray | None = None

    def conf_int(self, level: float = 0.95) -> tuple:
        """Two-sided normal confidence interval for the price."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        z = norm.ppf(0.5 + level / 2.0)
        return (self.price - z * self.stderr, self.price + z * self.stderr)

    def __repr__(self) -> str:
        return f"MCResult(price={self.price:.6g}, stderr={self.stderr:.3g}, n_paths={self.n_paths})"


def discount_factors(result, discount, T: float) -> np.ndarray | float:
    """Discount factor(s) to maturity ``T``.

    ``discount`` may be a continuously compounded rate, a :class:`ZeroCurve`, or
    ``None``; with ``None`` the pathwise ``"discount"`` state of the result is used
    when the process provides one, otherwise payoffs are left undiscounted.
    """
    if discount is None:
        if "discount" in result.paths:
            return result["discount"][:, -1]
        return 1.0
    if isinstance(discount, ZeroCurve):
        return float(discount.discount(T))
    return float(np.exp(-float(discount) * T))


def _estimate(values: np.ndarray, antithetic: bool) -> np.ndarray:
    """Independent samples: pair averages under antithetic sampling."""
    if antithetic:
        half = values.shape[0] // 2
        return 0.5 * (values[:half] + values[half:])
    return values


def price_mc(
    process,
    payoff,
    T: float,
    *,
    discount=None,
    n_steps: int = 252,
    n_paths: int = 100_000,
    seed=None,
    antithetic: bool = False,
    control_variate=None,
    grid=None,
    keep_samples: bool = False,
) -> MCResult:
    """Price a payoff by Monte Carlo simulation of a process.

    Parameters
    ----------
    process : StochasticProcess
        Process to simulate (e.g. a risk-neutral :class:`~stochPathSim.processes.GBM`).
    payoff : callable
        Maps a :class:`~stochPathSim.core.SimulationResult` to undiscounted payoffs.
    T : float
        Maturity in years.
    discount : float, ZeroCurve or None
        Discounting: a continuously compounded rate, a curve, or ``None`` to use
        the process's pathwise ``"discount"`` state if present (undiscounted otherwise).
    n_steps, n_paths : int
        Time steps and paths.
    seed : int, optional
        Seed for reproducible results.
    antithetic : bool
        Antithetic variates; the standard error is computed on pair averages.
    control_variate : tuple (payoff, exact_value), optional
        A second payoff with known present value. Its discounted samples are
        used as a linear control with the variance-minimising coefficient.
    grid : TimeGrid, optional
        Custom simulation grid (overrides ``T``/``n_steps`` for the simulation).
    keep_samples : bool
        Store the samples in the result.
    """
    result = process.simulate(T, n_steps, n_paths, seed=seed, antithetic=antithetic, grid=grid)
    T_end = float(result.times[-1])
    df = discount_factors(result, discount, T_end)
    values = _estimate(df * np.asarray(payoff(result), dtype=float), antithetic)

    if control_variate is not None:
        cv_payoff, cv_exact = control_variate
        controls = _estimate(df * np.asarray(cv_payoff(result), dtype=float), antithetic)
        var_c = controls.var(ddof=1)
        if var_c > 0.0:
            beta = np.cov(values, controls, ddof=1)[0, 1] / var_c
            values = values - beta * (controls - float(cv_exact))

    n = values.shape[0]
    stderr = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return MCResult(float(values.mean()), stderr, int(result.n_paths), values if keep_samples else None)
