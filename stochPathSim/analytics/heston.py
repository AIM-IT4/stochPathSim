"""Semi-analytic Heston (1993) option prices."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def heston_char_func(u, S, T, r, q, v0, kappa, theta, xi, rho):
    """Characteristic function ``E[exp(i u ln S_T)]`` under the risk-neutral measure.

    Uses the "little Heston trap" formulation of Albrecher et al. (2007), which avoids
    the branch-cut discontinuity of the original formula.
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    edt = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((b - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    D = (b - d) / xi**2 * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(iu * (np.log(S) + (r - q) * T) + C + D * v0)


def heston_price(S, K, T, r, q, v0, kappa, theta, xi, rho, option_type: str = "call",
                 upper: float = 200.0) -> float:
    """European option price in the Heston model by Gil-Pelaez inversion.

    Parameters
    ----------
    S, K, T, r, q : float
        Spot, strike, expiry, risk-free rate and dividend yield.
    v0, kappa, theta, xi, rho : float
        Initial variance, mean-reversion speed, long-run variance, vol of vol and correlation.
    option_type : {"call", "put"}
    upper : float
        Upper limit of the Fourier integrals.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if T <= 0.0:
        intrinsic = S - K if option_type == "call" else K - S
        return max(intrinsic, 0.0)
    if xi <= 0.0:
        raise ValueError("xi must be positive; use bs_price for xi = 0")
    args = (S, T, r, q, v0, kappa, theta, xi, rho)
    lnk = np.log(K)
    fwd = S * np.exp((r - q) * T)

    def p1_integrand(u):
        return (np.exp(-1j * u * lnk) * heston_char_func(u - 1j, *args) / (1j * u * fwd)).real

    def p2_integrand(u):
        return (np.exp(-1j * u * lnk) * heston_char_func(u, *args) / (1j * u)).real

    p1 = 0.5 + quad(p1_integrand, 0.0, upper, limit=400)[0] / np.pi
    p2 = 0.5 + quad(p2_integrand, 0.0, upper, limit=400)[0] / np.pi
    call = S * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2
    if option_type == "call":
        return float(call)
    return float(call - S * np.exp(-q * T) + K * np.exp(-r * T))
