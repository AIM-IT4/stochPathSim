"""Hagan et al. (2002) SABR implied volatility."""

from __future__ import annotations

import numpy as np


def sabr_implied_vol(F, K, T, alpha, beta, rho, nu, shift: float = 0.0):
    """Lognormal (Black) implied volatility from Hagan's SABR expansion.

    Parameters
    ----------
    F, K : float or array_like
        Forward and strike.
    T : float
        Time to expiry.
    alpha, beta, rho, nu : float
        SABR parameters.
    shift : float
        Shift for the shifted-SABR variant: the formula is applied to ``F + shift`` and
        ``K + shift``, and the result is a shifted-lognormal volatility.
    """
    F = np.asarray(F, dtype=float) + shift
    K = np.asarray(K, dtype=float) + shift
    if np.any(F <= 0.0) or np.any(K <= 0.0):
        raise ValueError("forward and strike (plus shift) must be positive")
    omb = 1.0 - beta
    fk = F * K
    log_fk = np.log(F / K)
    fk_b = fk ** (0.5 * omb)
    denom = fk_b * (1.0 + omb**2 / 24.0 * log_fk**2 + omb**4 / 1920.0 * log_fk**4)
    z = nu / alpha * fk_b * log_fk
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
        zx = np.where(np.abs(z) < 1e-12, 1.0 - 0.5 * rho * z, z / x)
    correction = 1.0 + (omb**2 / 24.0 * alpha**2 / fk ** omb
                        + 0.25 * rho * beta * nu * alpha / fk_b
                        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2) * T
    vol = alpha / denom * zx * correction
    return vol.item() if vol.ndim == 0 else vol
