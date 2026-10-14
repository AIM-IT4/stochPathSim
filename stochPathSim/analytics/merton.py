"""Merton (1976) jump-diffusion option prices."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from .black_scholes import bs_price


def merton_price(S, K, T, r, sigma, lam, jump_mean, jump_std, option_type: str = "call",
                 q: float = 0.0, n_terms: int = 100) -> float:
    """European option price under Merton's jump diffusion with log-normal jumps.

    The price is a Poisson-weighted sum of Black-Scholes prices (Merton, 1976).

    Parameters
    ----------
    lam : float
        Jump intensity per year.
    jump_mean, jump_std : float
        Mean and standard deviation of the normal log-jump size.
    n_terms : int
        Number of terms kept in the series.
    """
    if T <= 0.0:
        return float(bs_price(S, K, T, r, sigma, option_type, q))
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    lam_p = lam * (1.0 + k)
    n = np.arange(n_terms)
    weights = np.exp(-lam_p * T + n * np.log(lam_p * T) - gammaln(n + 1)) if lam_p > 0 else (n == 0) * 1.0
    sig_n = np.sqrt(sigma**2 + n * jump_std**2 / T)
    r_n = r - lam * k + n * np.log1p(k) / T
    # Weights use lam' = lam (1 + k); together with discounting at r_n this equals
    # discounting at r with Poisson(lam) weights.
    prices = bs_price(S, K, T, r_n, sig_n, option_type, q)
    return float(np.sum(weights * prices))
