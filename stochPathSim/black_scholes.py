"""Version 1 interface, kept for backward compatibility.

Prefer :func:`stochPathSim.analytics.bs_price` and :func:`stochPathSim.pricing.price_mc`.
"""

from __future__ import annotations

from .analytics.black_scholes import bs_price
from .pricing.montecarlo import price_mc
from .pricing.payoffs import European
from .processes.gbm import GBM


class BlackScholes:
    """Black-Scholes prices for European options."""

    @staticmethod
    def option_price(S, K, T, r, sigma, option_type="call"):
        """Closed-form Black-Scholes price."""
        if option_type not in ("call", "put"):
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        return bs_price(S, K, T, r, sigma, option_type)

    @staticmethod
    def monte_carlo_simulation(S0, K, T, r, sigma, paths, n, option_type="call", seed=None):
        """Monte Carlo price with ``n`` time points (the payoff only needs the last)."""
        if option_type not in ("call", "put"):
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        result = price_mc(GBM(S0, r, sigma), European(K, option_type), T, discount=r,
                          n_steps=max(int(n) - 1, 1), n_paths=paths, seed=seed)
        return result.price
