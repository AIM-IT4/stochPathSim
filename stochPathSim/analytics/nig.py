"""European option pricing under exponential NIG using tilted NIG probabilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import norminvgauss

from .black_scholes import _sign


def nig_price(S, K, T, r, alpha, beta, delta, option_type: str = "call", q: float = 0.0) -> float:
    """European NIG call or put via SciPy's NIG CDF/survival function.

    Scalar inputs. ``alpha``, ``beta``, ``delta`` use the same per-year
    parameterisation as :class:`~stochPathSim.processes.NormalInverseGaussian`.
    Exponential tilting changes beta to beta+1 under the stock measure.
    Discounted stock and strike times the corresponding exercise probabilities
    give the price, without Monte Carlo error. SciPy evaluates the distribution
    functions numerically, so small numerical integration errors are possible.

    T and delta may be zero. S and K must be positive. The model's first-moment
    domain is required; a finite second moment is not necessary for pricing.
    """
    from ..processes.nig import NormalInverseGaussian

    w = _sign(option_type)
    if not np.all(np.isfinite([K, T])) or K <= 0 or T < 0:
        raise ValueError("K must be positive and T non-negative, both finite")
    model = NormalInverseGaussian(S, r, alpha, beta, delta, q=q)
    stock_pv, strike_pv = S * np.exp(-q * T), K * np.exp(-r * T)
    if T == 0 or delta == 0:
        return float(max(w * (stock_pv - strike_pv), 0.0))
    scale = delta * T
    threshold = np.log(K / S) - (r - q - model.compensator) * T
    original = norminvgauss(alpha * scale, beta * scale, scale=scale)
    tilted = norminvgauss(alpha * scale, (beta + 1.0) * scale, scale=scale)
    if option_type == "call":
        return float(stock_pv * tilted.sf(threshold) - strike_pv * original.sf(threshold))
    return float(strike_pv * original.cdf(threshold) - stock_pv * tilted.cdf(threshold))
