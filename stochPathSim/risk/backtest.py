"""Backtests for VaR and risk-factor simulation models."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(n_obs: int, n_exceed: int, level: float = 0.99) -> tuple:
    """Kupiec proportion-of-failures test.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    n_exceed : int
        Number of exceedances.
    level : float
        VaR confidence level; the expected exceedance rate is ``1 - level``.

    Returns
    -------
    (float, float)
        Likelihood-ratio statistic (chi-square, 1 dof) and p-value.
    """
    if n_obs <= 0 or not 0 <= n_exceed <= n_obs:
        raise ValueError("need n_obs > 0 and 0 <= n_exceed <= n_obs")
    p = 1.0 - level
    phat = n_exceed / n_obs
    ll0 = _xlogy(n_obs - n_exceed, 1 - p) + _xlogy(n_exceed, p)
    ll1 = _xlogy(n_obs - n_exceed, 1 - phat) + _xlogy(n_exceed, phat)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(exceed_indicator) -> tuple:
    """Christoffersen test that exceedances are independent over time.

    Returns the likelihood-ratio statistic (chi-square, 1 dof) and p-value.
    """
    x = np.asarray(exceed_indicator).astype(bool).astype(int)
    if x.size < 2:
        raise ValueError("need at least two observations")
    prev, curr = x[:-1], x[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def pit_values(simulated_distributions, realised) -> np.ndarray:
    """Probability integral transform of realised values.

    Parameters
    ----------
    simulated_distributions : array_like
        Simulated values, shape ``(n_dates, n_sims)``: one forecast distribution per date.
    realised : array_like
        Realised values, shape ``(n_dates,)``.

    Returns
    -------
    ndarray
        ``F_i(x_i)`` for each date, the share of simulations at or below the
        realisation. Uniform on [0, 1] when the model is correct.
    """
    sims = np.atleast_2d(np.asarray(simulated_distributions, dtype=float))
    x = np.asarray(realised, dtype=float).ravel()
    if sims.shape[0] != x.size:
        raise ValueError("need one simulated distribution per realised value")
    return (sims <= x[:, None]).mean(axis=1)


def _clip_pit(u):
    u = np.asarray(u, dtype=float).ravel()
    if u.size < 2:
        raise ValueError("need at least two PIT values")
    eps = 1e-10
    return np.clip(u, eps, 1 - eps)


def anderson_darling_uniform(pit) -> tuple:
    """Anderson-Darling test of uniformity for PIT values.

    Returns the statistic and an asymptotic p-value (Marsaglia & Marsaglia 2004).
    """
    u = np.sort(_clip_pit(pit))
    n = u.size
    i = np.arange(1, n + 1)
    a2 = float(-n - np.mean((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1]))))
    return a2, float(_ad_pvalue(a2))


def _ad_pvalue(z):
    # asymptotic distribution of A^2 for a fully specified continuous distribution
    if z <= 0:
        return 1.0
    if z < 2:
        cdf = (
            np.exp(-1.2337141 / z)
            / np.sqrt(z)
            * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z) * z) * z)
        )
    else:
        cdf = np.exp(
            -np.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z) * z)
        )
    return float(min(max(1.0 - cdf, 0.0), 1.0))


def cramer_von_mises_uniform(pit) -> tuple:
    """Cramer-von Mises test of uniformity for PIT values (statistic, p-value)."""
    res = stats.cramervonmises(_clip_pit(pit), "uniform")
    return float(res.statistic), float(res.pvalue)
