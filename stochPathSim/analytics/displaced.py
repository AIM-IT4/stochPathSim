"""European pricing and forward Greeks in the shifted-lognormal model."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .black_scholes import black76_price


def _validated(F, K, T, df, sigma, shift, option_type):
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    F, K, T, df, sigma, shift = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (F, K, T, df, sigma, shift)))
    if any(np.any(~np.isfinite(x)) for x in (F, K, T, df, sigma, shift)):
        raise ValueError("inputs must be finite")
    if np.any(F + shift <= 0.0):
        raise ValueError("F + shift must be positive")
    if np.any(T < 0.0) or np.any(df <= 0.0) or np.any(sigma < 0.0):
        raise ValueError("T and sigma must be non-negative; df must be positive")
    return F, K, T, df, sigma, shift


def _scalar_or_array(value):
    return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value


def displaced_diffusion_price(F, K, T, df, sigma, shift, option_type: str = "call"):
    """Discounted European option on a displaced-lognormal *forward*.

    S_T + shift is lognormal with initial value F + shift, zero drift,
    and volatility sigma. Thus the shifted-strike payoff reduces exactly
    to Black-76. Negative original forwards and strikes are supported.
    If K + shift <= 0, the option is always in (or out of) the money;
    the resulting price is its discounted forward intrinsic value.

    F is a forward at expiry, NOT a spot subject to risk-free drift.
    Supports broadcastable scalar and array inputs, including T=0 and sigma=0.
    """
    F, K, T, df, sigma, shift = _validated(F, K, T, df, sigma, shift, option_type)
    return black76_price(F + shift, K + shift, T, df, sigma, option_type)


def displaced_diffusion_greeks(F, K, T, df, sigma, shift, option_type: str = "call"):
    """Forward delta, gamma, vega and frozen-discount theta.

    Vega is per unit change in displaced *relative* volatility sigma;
    theta is -d(price)/dT with F and df held fixed. The function requires
    T>0 and sigma>0; for K + shift <= 0 the payoff is linear and the
    corresponding Greeks are exact (zero gamma, vega and theta).
    """
    F, K, T, df, sigma, shift = _validated(F, K, T, df, sigma, shift, option_type)
    if np.any(T <= 0.0) or np.any(sigma <= 0.0):
        raise ValueError("Greeks require T > 0 and sigma > 0")
    s, k = F + shift, K + shift
    w = 1 if option_type == "call" else -1
    delta = np.where(k <= 0, df if w == 1 else 0.0, 0.0).astype(float)
    gamma = np.zeros_like(s, dtype=float)
    vega = np.zeros_like(s, dtype=float)
    theta = np.zeros_like(s, dtype=float)
    live = k > 0
    if np.any(live):
        sq = np.sqrt(T[live])
        d1 = (np.log(s[live] / k[live]) + 0.5 * sigma[live]**2 * T[live]) / (sigma[live] * sq)
        pdf = norm.pdf(d1)
        delta[live] = df[live] * w * norm.cdf(w * d1)
        gamma[live] = df[live] * pdf / (s[live] * sigma[live] * sq)
        vega[live] = df[live] * s[live] * sq * pdf
        theta[live] = -df[live] * s[live] * sigma[live] * pdf / (2 * sq)
    return {name: _scalar_or_array(v) for name, v in
            dict(delta=delta, gamma=gamma, vega=vega, theta=theta).items()}
