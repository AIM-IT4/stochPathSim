"""Closed-form bond and option prices for short-rate models."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..curves import ZeroCurve


def _curve(curve) -> ZeroCurve:
    return curve if isinstance(curve, ZeroCurve) else ZeroCurve.flat(float(curve))


def _b(k, tau):
    return -np.expm1(-k * tau) / k


def _sign(option_type: str) -> int:
    if option_type == "call":
        return 1
    if option_type == "put":
        return -1
    raise ValueError("option_type must be 'call' or 'put'")


def _bond_option(p_bond, p_expiry, strike, sigma_p, option_type):
    """Black-type formula for an option on a zero-coupon bond."""
    w = _sign(option_type)
    sigma_p = np.asarray(sigma_p, dtype=float)
    h = np.log(p_bond / (strike * p_expiry)) / sigma_p + 0.5 * sigma_p
    return w * (p_bond * norm.cdf(w * h) - strike * p_expiry * norm.cdf(w * (h - sigma_p)))


def vasicek_zcb(r, t, T, kappa, theta, sigma):
    """Vasicek zero-coupon bond price ``P(t, T)`` given the short rate ``r`` at ``t``."""
    tau = np.asarray(T, dtype=float) - np.asarray(t, dtype=float)
    b = _b(kappa, tau)
    log_a = (theta - sigma**2 / (2 * kappa**2)) * (b - tau) - sigma**2 * b**2 / (4 * kappa)
    return np.exp(log_a - b * np.asarray(r, dtype=float))


def vasicek_zcb_option(r0, T_option, T_bond, strike, kappa, theta, sigma, option_type="call"):
    """European option expiring at ``T_option`` on a zero-coupon bond maturing at ``T_bond``."""
    p_expiry = vasicek_zcb(r0, 0.0, T_option, kappa, theta, sigma)
    p_bond = vasicek_zcb(r0, 0.0, T_bond, kappa, theta, sigma)
    sigma_p = (sigma * _b(kappa, np.asarray(T_bond) - np.asarray(T_option))
               * np.sqrt(-np.expm1(-2 * kappa * np.asarray(T_option)) / (2 * kappa)))
    return _bond_option(p_bond, p_expiry, strike, sigma_p, option_type)


def cir_zcb(r, t, T, kappa, theta, sigma):
    """Cox-Ingersoll-Ross zero-coupon bond price ``P(t, T)`` given ``r`` at ``t``."""
    tau = np.asarray(T, dtype=float) - np.asarray(t, dtype=float)
    h = np.sqrt(kappa**2 + 2 * sigma**2)
    em1 = np.expm1(h * tau)
    denom = 2 * h + (kappa + h) * em1
    log_a = 2 * kappa * theta / sigma**2 * (np.log(2 * h) + 0.5 * (kappa + h) * tau - np.log(denom))
    b = 2 * em1 / denom
    return np.exp(log_a - b * np.asarray(r, dtype=float))


def hull_white_zcb(r, t, T, a, sigma, curve):
    """Hull-White zero-coupon bond price ``P(t, T)`` given the short rate ``r`` at ``t``.

    ``curve`` is the initial :class:`~stochPathSim.curves.ZeroCurve` (or a flat rate).
    """
    curve = _curve(curve)
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    b = _b(a, T - t)
    log_a = (np.log(curve.discount(T) / curve.discount(t)) + b * curve.forward_rate(t)
             - sigma**2 / (4 * a) * -np.expm1(-2 * a * t) * b**2)
    return np.exp(log_a - b * np.asarray(r, dtype=float))


def hull_white_zcb_option(T_option, T_bond, strike, a, sigma, curve, option_type="call"):
    """Hull-White European option on a zero-coupon bond, valued at time 0."""
    curve = _curve(curve)
    p_expiry = curve.discount(T_option)
    p_bond = curve.discount(T_bond)
    sigma_p = (sigma * _b(a, np.asarray(T_bond) - np.asarray(T_option))
               * np.sqrt(-np.expm1(-2 * a * np.asarray(T_option)) / (2 * a)))
    return _bond_option(p_bond, p_expiry, strike, sigma_p, option_type)


def hull_white_caplet(T_reset, T_pay, strike, a, sigma, curve, notional=1.0, floorlet=False):
    """Hull-White caplet (or floorlet) on the simply compounded rate over ``[T_reset, T_pay]``.

    A caplet is ``1 + K tau`` puts on the ``T_pay`` bond struck at ``1 / (1 + K tau)``.
    """
    tau = np.asarray(T_pay, dtype=float) - np.asarray(T_reset, dtype=float)
    factor = 1.0 + strike * tau
    option_type = "call" if floorlet else "put"
    return notional * factor * hull_white_zcb_option(T_reset, T_pay, 1.0 / factor, a, sigma, curve,
                                                     option_type)


def g2pp_integral_variance(tau, a, b, sigma, eta, rho):
    """Variance of ``int_t^{t+tau} (x_s + y_s) ds`` given the factors at ``t``."""
    tau = np.asarray(tau, dtype=float)
    vx = sigma**2 / a**2 * (tau - 2 * _b(a, tau) + _b(2 * a, tau))
    vy = eta**2 / b**2 * (tau - 2 * _b(b, tau) + _b(2 * b, tau))
    vxy = 2 * rho * sigma * eta / (a * b) * (tau - _b(a, tau) - _b(b, tau) + _b(a + b, tau))
    return vx + vy + vxy


def g2pp_zcb(x, y, t, T, a, b, sigma, eta, rho, curve):
    """G2++ zero-coupon bond price ``P(t, T)`` given the factors ``x`` and ``y`` at ``t``.

    At ``t = 0`` with ``x = y = 0`` this equals ``curve.discount(T)``.
    """
    curve = _curve(curve)
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    v = lambda tau: g2pp_integral_variance(tau, a, b, sigma, eta, rho)  # noqa: E731
    log_a = np.log(curve.discount(T) / curve.discount(t)) + 0.5 * (v(T - t) - v(T) + v(t))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.exp(log_a - _b(a, T - t) * x - _b(b, T - t) * y)


__all__ = [
    "vasicek_zcb",
    "vasicek_zcb_option",
    "cir_zcb",
    "hull_white_zcb",
    "hull_white_zcb_option",
    "hull_white_caplet",
    "g2pp_integral_variance",
    "g2pp_zcb",
]
