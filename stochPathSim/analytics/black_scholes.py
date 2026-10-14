"""Black-Scholes, Black-76 and Bachelier closed forms."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _sign(option_type: str) -> float:
    if option_type == "call":
        return 1.0
    if option_type == "put":
        return -1.0
    raise ValueError("option_type must be 'call' or 'put'")


def _scalar_or_array(x):
    return x.item() if isinstance(x, np.ndarray) and x.ndim == 0 else x


def black76_price(F, K, T, df, sigma, option_type: str = "call"):
    """Black-76 price of a European option on a forward.

    Parameters
    ----------
    F, K : float or array_like
        Forward and strike.
    T : float or array_like
        Time to expiry in years.
    df : float or array_like
        Discount factor to the payment date.
    sigma : float or array_like
        Lognormal volatility.
    option_type : {"call", "put"}
    """
    w = _sign(option_type)
    F, K, T, df, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (F, K, T, df, sigma)))
    sd = sigma * np.sqrt(np.maximum(T, 0.0))
    intrinsic = df * np.maximum(w * (F - K), 0.0)
    live = (sd > 0.0) & (F > 0.0) & (K > 0.0)
    out = np.array(intrinsic, dtype=float)
    if np.any(live):
        f, k, s, d = F[live], K[live], sd[live], df[live]
        d1 = np.log(f / k) / s + 0.5 * s
        d2 = d1 - s
        out[live] = d * w * (f * norm.cdf(w * d1) - k * norm.cdf(w * d2))
    return _scalar_or_array(out)


def bs_price(S, K, T, r, sigma, option_type: str = "call", q=0.0):
    """Black-Scholes price of a European option with continuous dividend yield ``q``."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    F = S * np.exp((r - q) * T)
    return black76_price(F, K, T, np.exp(-r * T), sigma, option_type)


def bs_greeks(S, K, T, r, sigma, option_type: str = "call", q=0.0) -> dict:
    """Black-Scholes greeks: ``delta``, ``gamma``, ``vega``, ``theta`` (per year) and ``rho``.

    Vega and rho are per unit change (not per 1%).
    """
    w = _sign(option_type)
    S, K, T, r, sigma, q = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q)))
    if np.any(T <= 0.0) or np.any(sigma <= 0.0):
        raise ValueError("greeks need T > 0 and sigma > 0")
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    pdf = norm.pdf(d1)
    delta = w * dq * norm.cdf(w * d1)
    gamma = dq * pdf / (S * sigma * sq)
    vega = S * dq * pdf * sq
    theta = (-S * dq * pdf * sigma / (2 * sq) - w * r * K * dr * norm.cdf(w * d2)
             + w * q * S * dq * norm.cdf(w * d1))
    rho = w * K * T * dr * norm.cdf(w * d2)
    return {k: _scalar_or_array(v) for k, v in
            dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho).items()}


def _implied(price_fn, price, lower, upper, lo=1e-6, hi=5.0):
    if not (lower - 1e-12 <= price <= upper + 1e-12):
        return np.nan
    f_lo, f_hi = price_fn(lo) - price, price_fn(hi) - price
    if f_lo > 0:
        return lo if abs(f_lo) < 1e-10 else np.nan
    if f_hi < 0:
        return np.nan
    return brentq(lambda s: price_fn(s) - price, lo, hi, xtol=1e-12, maxiter=200)


def bs_implied_vol(price, S, K, T, r, option_type: str = "call", q=0.0):
    """Black-Scholes implied volatility by Brent's method on ``[1e-6, 5]``.

    Returns ``nan`` for prices outside the no-arbitrage bounds. Accepts arrays.
    """
    w = _sign(option_type)
    args = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (price, S, K, T, r, q)))
    out = np.empty(args[0].shape)
    for idx in np.ndindex(out.shape):
        p, s, k, t, rr, qq = (a[idx] for a in args)
        if t <= 0:
            out[idx] = np.nan
            continue
        F, df = s * np.exp((rr - qq) * t), np.exp(-rr * t)
        lower = df * max(w * (F - k), 0.0)
        upper = df * (F if w > 0 else k)
        out[idx] = _implied(lambda v, F=F, k=k, t=t, df=df: black76_price(F, k, t, df, v, option_type),
                            p, lower, upper)
    return _scalar_or_array(out)


def bachelier_price(F, K, T, df, sigma_n, option_type: str = "call"):
    """Bachelier (normal model) price of a European option on a forward."""
    w = _sign(option_type)
    F, K, T, df, sigma_n = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (F, K, T, df, sigma_n)))
    sd = sigma_n * np.sqrt(np.maximum(T, 0.0))
    out = np.array(df * np.maximum(w * (F - K), 0.0), dtype=float)
    live = sd > 0.0
    if np.any(live):
        s = sd[live]
        d = w * (F[live] - K[live]) / s
        out[live] = df[live] * s * (d * norm.cdf(d) + norm.pdf(d))
    return _scalar_or_array(out)


def bachelier_implied_vol(price, F, K, T, df, option_type: str = "call"):
    """Normal implied volatility; ``nan`` outside the no-arbitrage bounds."""
    w = _sign(option_type)
    args = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (price, F, K, T, df)))
    out = np.empty(args[0].shape)
    for idx in np.ndindex(out.shape):
        p, f, k, t, d = (a[idx] for a in args)
        if t <= 0:
            out[idx] = np.nan
            continue
        lower = d * max(w * (f - k), 0.0)
        hi = 10.0 * max(abs(f), abs(k), 1e-2) / np.sqrt(t) + 1.0
        out[idx] = _implied(lambda v, f=f, k=k, t=t, d=d: bachelier_price(f, k, t, d, v, option_type),
                            p, lower, np.inf, lo=1e-12, hi=hi)
    return _scalar_or_array(out)
