"""Value at risk and expected shortfall.

Losses are reported as positive numbers: for P&L ``x``, the loss is ``-x``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def var_es(pnl, level: float = 0.99, method: str = "historical") -> tuple:
    """VaR and expected shortfall of a P&L sample.

    Parameters
    ----------
    pnl : array_like
        Profit-and-loss observations or simulations.
    level : float
        Confidence level, e.g. 0.99.
    method : {"historical", "parametric", "cornish_fisher"}
        Empirical quantile, normal approximation, or normal with a
        Cornish-Fisher skew/kurtosis adjustment (ES from the empirical tail
        beyond that VaR for Cornish-Fisher).

    Returns
    -------
    (float, float)
        ``(VaR, ES)``, losses positive.
    """
    x = np.asarray(pnl, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("need at least two P&L values")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    losses = -x
    alpha = level
    if method == "historical":
        var = float(np.quantile(losses, alpha))
        tail = losses[losses >= var]
        es = float(tail.mean()) if tail.size else var
    elif method == "parametric":
        mu, sd = losses.mean(), losses.std(ddof=1)
        z = stats.norm.ppf(alpha)
        var = float(mu + sd * z)
        es = float(mu + sd * stats.norm.pdf(z) / (1.0 - alpha))
    elif method == "cornish_fisher":
        mu, sd = losses.mean(), losses.std(ddof=1)
        s = stats.skew(losses)
        k = stats.kurtosis(losses)  # excess kurtosis
        z = stats.norm.ppf(alpha)
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        var = float(mu + sd * z_cf)
        tail = losses[losses >= var]
        es = float(tail.mean()) if tail.size else var
    else:
        raise ValueError("method must be 'historical', 'parametric' or 'cornish_fisher'")
    return var, es


def monte_carlo_var(terminal_values, initial_value: float, level: float = 0.99) -> tuple:
    """VaR and ES of ``terminal_values - initial_value`` from simulated values."""
    pnl = np.asarray(terminal_values, dtype=float) - float(initial_value)
    return var_es(pnl, level=level, method="historical")
