"""Maximum-likelihood calibration of diffusion models to time series."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats


@dataclass
class FitResult:
    """Estimated parameters and the log-likelihood at the optimum."""

    params: dict
    log_likelihood: float
    n_obs: int
    info: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.params[key]

    @property
    def aic(self) -> float:
        return 2 * len(self.params) - 2 * self.log_likelihood


def _series(x, min_len=3):
    x = np.asarray(x, dtype=float).ravel()
    if x.size < min_len:
        raise ValueError(f"need at least {min_len} observations")
    return x


def fit_gbm(prices, dt: float) -> FitResult:
    """Fit ``dS = mu S dt + sigma S dW`` by MLE on log returns."""
    s = _series(prices)
    if np.any(s <= 0):
        raise ValueError("prices must be positive")
    r = np.diff(np.log(s))
    m, v = r.mean(), r.var()  # MLE variance (ddof=0)
    sigma = np.sqrt(v / dt)
    mu = m / dt + 0.5 * sigma**2
    ll = float(stats.norm.logpdf(r, m, np.sqrt(v)).sum())
    return FitResult({"mu": float(mu), "sigma": float(sigma)}, ll, r.size)


def fit_ou(x, dt: float) -> FitResult:
    """Exact MLE of ``dX = kappa (theta - X) dt + sigma dW`` (also Vasicek).

    Uses the AR(1) representation ``X_{t+dt} = a + b X_t + e``.
    """
    x = _series(x)
    x0, x1 = x[:-1], x[1:]
    n = x0.size
    b, a = np.polyfit(x0, x1, 1)
    if not 0.0 < b < 1.0:
        raise ValueError("series shows no mean reversion (AR(1) slope outside (0, 1))")
    resid = x1 - (a + b * x0)
    s2 = resid.var()
    kappa = -np.log(b) / dt
    theta = a / (1.0 - b)
    sigma = np.sqrt(s2 * 2.0 * kappa / (1.0 - b**2))
    ll = float(stats.norm.logpdf(resid, 0.0, np.sqrt(s2)).sum())
    return FitResult({"kappa": float(kappa), "theta": float(theta), "sigma": float(sigma)}, ll, n)


def _cir_loglik(params, r0, r1, dt):
    kappa, theta, sigma = params
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return -np.inf
    c = 2.0 * kappa / (sigma**2 * (1.0 - np.exp(-kappa * dt)))
    df = 4.0 * kappa * theta / sigma**2
    nc = 2.0 * c * r0 * np.exp(-kappa * dt)
    ll = stats.ncx2.logpdf(2.0 * c * r1, df, nc) + np.log(2.0 * c)
    return float(np.sum(ll))


def fit_cir(r, dt: float) -> FitResult:
    """MLE of ``dr = kappa (theta - r) dt + sigma sqrt(r) dW`` with the exact
    noncentral chi-square transition density, started from OLS estimates."""
    r = _series(r)
    if np.any(r <= 0):
        raise ValueError("rates must be positive for the CIR likelihood")
    r0, r1 = r[:-1], r[1:]
    # OLS on the discretised SDE scaled by sqrt(r)
    y = (r1 - r0) / np.sqrt(r0)
    X = np.column_stack([dt / np.sqrt(r0), dt * np.sqrt(r0)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    kappa0 = max(-beta[1], 1e-3)
    theta0 = max(beta[0] / kappa0, 1e-4)
    sigma0 = max(np.std(y - X @ beta) / np.sqrt(dt), 1e-4)
    start = np.array([kappa0, theta0, sigma0])
    res = optimize.minimize(
        lambda p: -_cir_loglik(p, r0, r1, dt),
        start,
        method="L-BFGS-B",
        bounds=[(1e-6, 50.0), (1e-6, 1.0), (1e-6, 5.0)],
    )
    kappa, theta, sigma = res.x
    return FitResult(
        {"kappa": float(kappa), "theta": float(theta), "sigma": float(sigma)},
        float(-res.fun),
        r0.size,
        info={
            "start": dict(zip(("kappa", "theta", "sigma"), map(float, start), strict=True)),
            "converged": bool(res.success),
            "feller": bool(2 * kappa * theta >= sigma**2),
        },
    )
