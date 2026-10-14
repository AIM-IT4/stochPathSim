import numpy as np
import pytest

from stochPathSim.calibration import fit_cir, fit_gbm, fit_ou
from stochPathSim.processes import GBM


def _ou_path(kappa, theta, sigma, x0, dt, n, rng):
    x = np.empty(n + 1)
    x[0] = x0
    b = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1 - b**2) / (2 * kappa))
    for i in range(n):
        x[i + 1] = theta + (x[i] - theta) * b + sd * rng.standard_normal()
    return x


def _cir_path(kappa, theta, sigma, r0, dt, n, rng):
    r = np.empty(n + 1)
    r[0] = r0
    c = sigma**2 * (1 - np.exp(-kappa * dt)) / (4 * kappa)
    df = 4 * kappa * theta / sigma**2
    for i in range(n):
        nc = r[i] * np.exp(-kappa * dt) / c
        r[i + 1] = c * rng.noncentral_chisquare(df, nc)
    return r


def test_fit_gbm_recovers_sigma():
    dt = 1 / 252
    res = GBM(100.0, 0.08, 0.25).simulate(T=20.0, n_steps=5040, n_paths=1, seed=3)
    fit = fit_gbm(res.get()[0], dt)
    assert fit["sigma"] == pytest.approx(0.25, rel=0.03)
    assert fit["mu"] == pytest.approx(0.08, abs=0.12)  # drift is notoriously hard to estimate
    assert fit.n_obs == 5040
    assert np.isfinite(fit.log_likelihood) and np.isfinite(fit.aic)


def test_fit_ou_recovers_parameters():
    rng = np.random.default_rng(10)
    x = _ou_path(2.0, 0.05, 0.02, 0.03, 1 / 52, 20_000, rng)
    fit = fit_ou(x, 1 / 52)
    assert fit["kappa"] == pytest.approx(2.0, rel=0.1)
    assert fit["theta"] == pytest.approx(0.05, abs=0.002)
    assert fit["sigma"] == pytest.approx(0.02, rel=0.03)


def test_fit_ou_rejects_random_walk():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        fit_ou(np.cumsum(rng.standard_normal(2000)) + np.arange(2000) * 0.5, 1.0)


def test_fit_cir_recovers_parameters():
    rng = np.random.default_rng(12)
    r = _cir_path(1.5, 0.04, 0.1, 0.03, 1 / 52, 15_000, rng)
    fit = fit_cir(r, 1 / 52)
    assert fit["kappa"] == pytest.approx(1.5, rel=0.15)
    assert fit["theta"] == pytest.approx(0.04, rel=0.05)
    assert fit["sigma"] == pytest.approx(0.1, rel=0.03)
    assert fit.info["converged"]
    assert fit.info["feller"]


def test_fit_inputs_validated():
    with pytest.raises(ValueError):
        fit_gbm([100.0, -1.0, 2.0], 1.0)
    with pytest.raises(ValueError):
        fit_cir([0.01, 0.0, 0.02], 1.0)
    with pytest.raises(ValueError):
        fit_ou([1.0, 2.0], 1.0)
