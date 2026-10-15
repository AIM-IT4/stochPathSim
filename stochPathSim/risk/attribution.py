"""Euler allocation of delta-normal portfolio VaR and expected shortfall."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class PortfolioRisk:
    """Normal VaR/ES and risk contributions in portfolio P&L currency units.

    Marginal arrays are derivatives with respect to position weights;
    component arrays multiply marginal derivatives by those weights.
    Component contributions sum to total risk, including expected P&L.
    Negative VaR/contributions are possible for profitable or hedged books.
    """

    var: float
    es: float
    volatility: float
    expected_pnl: float
    marginal_var: np.ndarray
    component_var: np.ndarray
    marginal_es: np.ndarray
    component_es: np.ndarray
    level: float
    horizon: float


def normal_portfolio_risk(weights, covariance, mean=None, *,
                          level: float = 0.99, horizon: float = 1.0) -> PortfolioRisk:
    """Delta-normal portfolio VaR/ES with additive Euler risk attribution.

    Positions weights multiply one-period asset P&Ls. mean and covariance
    describe annualised (or other unit-time) P&L mean and covariance,
    respectively. Both scale linearly with horizon. Correlated assets and
    negative portfolio weights are supported.

    Portfolio loss is the negative of P&L. VaR = z*sd - expected_pnl
    and ES = phi(z)/(1-level)*sd - expected_pnl. Components add to total
    VaR/ES by positive homogeneity in positions.

    A zero-volatility portfolio has zero covariance risk and only
    deterministic mean-based contributions. Parameter uncertainty and
    nonnormal loss tails are not included.
    """
    w = np.asarray(weights, dtype=float)
    c = np.asarray(covariance, dtype=float)
    if w.ndim != 1 or w.size == 0 or c.shape != (w.size, w.size):
        raise ValueError("weights must be 1D and covariance must be square with matching dimension")
    m = np.zeros_like(w) if mean is None else np.asarray(mean, dtype=float)
    if m.shape != w.shape:
        raise ValueError("mean must have same shape as weights")
    if not all(np.all(np.isfinite(a)) for a in (w, c, m)):
        raise ValueError("all portfolio inputs must be finite")
    if not np.allclose(c, c.T, atol=1e-12, rtol=1e-12):
        raise ValueError("covariance must be symmetric")
    if np.linalg.eigvalsh(c).min() < -1e-10 * max(1.0, np.linalg.norm(c, ord=2)):
        raise ValueError("covariance must be positive semidefinite")
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must lie in (0, 1)")
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be positive and finite")
    c = (c + c.T)/2
    v = float(w @ c @ w)
    if v < -1e-10:
        raise ValueError("portfolio variance cannot be negative")
    std = np.sqrt(max(v, 0.0) * horizon)
    pnl = float(horizon * (w @ m))
    z = norm.ppf(level)
    coeff_es = norm.pdf(z)/(1 - level)
    gradient_sigma = horizon * (c @ w) / std if std > 0 else np.zeros_like(w)
    marg_var = z*gradient_sigma - horizon*m
    marg_es = coeff_es*gradient_sigma - horizon*m
    return PortfolioRisk(
        var=float(z*std - pnl),
        es=float(coeff_es*std - pnl),
        volatility=float(std),
        expected_pnl=pnl,
        marginal_var=marg_var,
        component_var=w*marg_var,
        marginal_es=marg_es,
        component_es=w*marg_es,
        level=float(level),
        horizon=float(horizon),
    )
