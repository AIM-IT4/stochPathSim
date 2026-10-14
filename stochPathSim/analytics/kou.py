"""Kou (2002) double-exponential jump-diffusion option prices."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def kou_char_func(u, S, T, r, q, sigma, lam, p_up, eta_up, eta_down):
    """Characteristic function of ``ln S_T`` under the risk-neutral measure."""
    u = np.asarray(u, dtype=complex)
    zeta = p_up * eta_up / (eta_up - 1.0) + (1.0 - p_up) * eta_down / (eta_down + 1.0) - 1.0
    phi_y = p_up * eta_up / (eta_up - 1j * u) + (1.0 - p_up) * eta_down / (eta_down + 1j * u)
    drift = np.log(S) + (r - q - 0.5 * sigma**2 - lam * zeta) * T
    return np.exp(1j * u * drift - 0.5 * sigma**2 * u**2 * T + lam * T * (phi_y - 1.0))


def kou_price(S, K, T, r, sigma, lam, p_up, eta_up, eta_down, option_type: str = "call",
              q: float = 0.0, upper: float = 500.0) -> float:
    """European option price under the Kou model by Gil-Pelaez Fourier inversion.

    Parameters
    ----------
    lam, p_up, eta_up, eta_down : float
        Jump intensity, probability of an upward jump and the up/down exponential rates
        (``eta_up > 1``).
    upper : float
        Upper limit of the Fourier integrals.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if not eta_up > 1.0:
        raise ValueError("eta_up must exceed 1")
    if T <= 0.0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    args = (S, T, r, q, sigma, lam, p_up, eta_up, eta_down)
    log_k = np.log(K)
    forward = S * np.exp((r - q) * T)

    def p2(u):
        return (np.exp(-1j * u * log_k) * kou_char_func(u, *args) / (1j * u)).real

    def p1(u):
        return (np.exp(-1j * u * log_k) * kou_char_func(u - 1j, *args) / (1j * u * forward)).real

    prob2 = 0.5 + quad(p2, 1e-10, upper, limit=500)[0] / np.pi
    prob1 = 0.5 + quad(p1, 1e-10, upper, limit=500)[0] / np.pi
    call = S * np.exp(-q * T) * prob1 - K * np.exp(-r * T) * prob2
    if option_type == "call":
        return float(call)
    return float(call - S * np.exp(-q * T) + K * np.exp(-r * T))
