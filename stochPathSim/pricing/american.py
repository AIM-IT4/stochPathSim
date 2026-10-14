"""American and Bermudan options by least-squares Monte Carlo."""

from __future__ import annotations

import numpy as np
from numpy.polynomial import laguerre

from .montecarlo import MCResult


def _basis(x: np.ndarray, basis: str, degree: int) -> np.ndarray:
    if basis == "laguerre":
        return laguerre.lagvander(x, degree)
    if basis == "poly":
        return np.vander(x, degree + 1, increasing=True)
    raise ValueError(f"basis must be 'laguerre' or 'poly', got {basis!r}")


def longstaff_schwartz(
    process,
    payoff_fn,
    T: float,
    *,
    r: float,
    n_steps: int = 50,
    n_paths: int = 100_000,
    basis: str = "laguerre",
    degree: int = 3,
    seed=None,
    antithetic: bool = False,
    state: str | None = None,
    grid=None,
) -> MCResult:
    """Price an American (Bermudan) option with the Longstaff-Schwartz algorithm.

    Exercise is allowed at every grid time after ``t = 0``; the continuation value
    is regressed on in-the-money paths only.

    Parameters
    ----------
    process : StochasticProcess
        Risk-neutral process for the underlying.
    payoff_fn : callable
        Intrinsic value as a function of the underlying, e.g.
        ``lambda s: np.maximum(K - s, 0)``.
    T : float
        Maturity in years.
    r : float
        Continuously compounded discount rate.
    n_steps : int
        Number of exercise dates (equal steps).
    n_paths : int
        Number of paths.
    basis : {"laguerre", "poly"}
        Regression basis, evaluated on the spot scaled by its initial value.
    degree : int
        Degree of the regression basis.
    seed, antithetic, grid
        As in :meth:`StochasticProcess.simulate`.
    state : str, optional
        State variable to exercise on (default: primary state).
    """
    if degree < 1:
        raise ValueError("degree must be at least 1")
    result = process.simulate(T, n_steps, n_paths, seed=seed, antithetic=antithetic, grid=grid)
    spots = result.get(state)
    times = result.times
    scale = float(np.mean(spots[:, 0]))

    cash = np.asarray(payoff_fn(spots[:, -1]), dtype=float)
    for i in range(times.size - 2, 0, -1):
        cash *= np.exp(-r * (times[i + 1] - times[i]))
        intrinsic = np.asarray(payoff_fn(spots[:, i]), dtype=float)
        itm = intrinsic > 0.0
        if np.count_nonzero(itm) > degree + 1:
            X = _basis(spots[itm, i] / scale, basis, degree)
            coef, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
            exercise = intrinsic[itm] > X @ coef
            idx = np.flatnonzero(itm)[exercise]
            cash[idx] = intrinsic[idx]
    cash *= np.exp(-r * (times[1] - times[0]))

    if antithetic:
        half = cash.shape[0] // 2
        samples = 0.5 * (cash[:half] + cash[half:])
    else:
        samples = cash
    price = float(samples.mean())
    stderr = float(samples.std(ddof=1) / np.sqrt(samples.shape[0]))
    immediate = float(np.mean(payoff_fn(spots[:, 0])))
    if immediate > price:
        return MCResult(immediate, 0.0, int(result.n_paths))
    return MCResult(price, stderr, int(result.n_paths))
