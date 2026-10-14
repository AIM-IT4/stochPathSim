"""Binomial trees for European and American options."""

from __future__ import annotations

import numpy as np


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    *,
    american: bool = False,
    q: float = 0.0,
    n_steps: int = 500,
) -> float:
    """Cox-Ross-Rubinstein binomial price of a vanilla option.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, maturity (years), continuously compounded rate and volatility.
    option_type : {"call", "put"}
    american : bool
        Allow early exercise at every node.
    q : float
        Continuous dividend yield.
    n_steps : int
        Number of tree steps.
    """
    option_type = str(option_type).lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        raise ValueError("S, K, T and sigma must be positive")
    n = int(n_steps)
    if n < 1:
        raise ValueError("n_steps must be at least 1")

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("no-arbitrage probability outside (0, 1); increase n_steps")
    disc = np.exp(-r * dt)
    sign = 1.0 if option_type == "call" else -1.0

    j = np.arange(n + 1)
    values = np.maximum(sign * (S * u ** (n - 2 * j) - K), 0.0)
    for i in range(n - 1, -1, -1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            j = np.arange(i + 1)
            values = np.maximum(values, sign * (S * u ** (i - 2 * j) - K))
    return float(values[0])
