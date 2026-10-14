import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from stochPathSim.analytics.black_scholes import bs_implied_vol, bs_price
from stochPathSim.analytics.kou import kou_price
from stochPathSim.analytics.rates import cir_zcb
from stochPathSim.core import TimeGrid
from stochPathSim.processes.commodity import SchwartzOneFactor, SchwartzSmith
from stochPathSim.processes.credit import CIRIntensity, ConstantIntensity
from stochPathSim.processes.jumps import KouJumpDiffusion
from stochPathSim.processes.regime import RegimeSwitchingGBM
from stochPathSim.processes.rough import RoughBergomi, _volterra_cov


def _within(sample, expected, k=4.0):
    sample = np.asarray(sample, dtype=float)
    stderr = sample.std(ddof=1) / np.sqrt(sample.size)
    assert abs(sample.mean() - expected) < k * stderr + 1e-12, (sample.mean(), expected, stderr)


# ---------------------------------------------------------------- rough Bergomi

def test_volterra_covariance_matches_quadrature():
    h = 0.1
    a = h - 0.5
    for u, v in [(0.3, 0.3), (0.2, 0.7), (0.5, 0.51), (1.0, 2.0)]:
        ref = 2 * h * quad(lambda s, u=u, v=v: (u - s) ** a * (v - s) ** a, 0.0, min(u, v), limit=200)[0]
        got = _volterra_cov(np.array([u]), np.array([v]), h)[0]
        assert got == pytest.approx(ref, rel=1e-6)


def test_rough_bergomi_moments():
    model = RoughBergomi(s0=100.0, mu=0.02, xi0=0.04, eta=1.9, hurst=0.1, rho=-0.9)
    res = model.simulate(T=1.0, n_steps=100, n_paths=40_000, seed=1)
    assert np.all(res["spot"][:, 0] == 100.0)
    _within(res["variance"][:, 50], 0.04)
    _within(res["variance"][:, -1], 0.04)
    _within(res.terminal(), model.mean(1.0))


def test_rough_bergomi_negative_skew_steepens_at_short_maturity():
    model = RoughBergomi(s0=1.0, mu=0.0, xi0=0.04, eta=1.9, hurst=0.1, rho=-0.9)

    def skew(T):
        s = model.simulate(T=T, n_steps=64, n_paths=60_000, seed=2, antithetic=True).terminal()
        k1, k2 = np.exp(-0.05 * np.sqrt(T)), np.exp(0.05 * np.sqrt(T))
        vols = [bs_implied_vol(np.maximum(s - k, 0).mean(), 1.0, k, T, 0.0) for k in (k1, k2)]
        return (vols[1] - vols[0]) / (np.log(k2) - np.log(k1))

    short, long_ = skew(0.1), skew(1.0)
    assert short < 0.0 and long_ < 0.0
    assert abs(short) > abs(long_)


def test_rough_bergomi_validation():
    with pytest.raises(ValueError):
        RoughBergomi(100, 0, 0.04, 1.9, 0.7, -0.9)


# ---------------------------------------------------------------- Kou

KOU = dict(sigma=0.16, lam=1.0, p_up=0.4, eta_up=10.0, eta_down=5.0)


def test_kou_mean():
    model = KouJumpDiffusion(100.0, 0.05, q=0.01, **KOU)
    s = model.simulate(T=1.0, n_steps=4, n_paths=200_000, seed=3).terminal()
    _within(s, model.mean(1.0))


def test_kou_price_reduces_to_black_scholes():
    p = kou_price(100, 105, 0.5, 0.03, 0.2, 0.0, 0.5, 10.0, 5.0)
    assert p == pytest.approx(bs_price(100, 105, 0.5, 0.03, 0.2), abs=1e-7)


def test_kou_put_call_parity():
    c = kou_price(100, 95, 1.0, 0.05, **KOU, option_type="call", q=0.02)
    p = kou_price(100, 95, 1.0, 0.05, **KOU, option_type="put", q=0.02)
    assert c - p == pytest.approx(100 * np.exp(-0.02) - 95 * np.exp(-0.05), abs=1e-7)


@pytest.mark.parametrize("K", [90.0, 100.0, 115.0])
def test_kou_monte_carlo_matches_closed_form(K):
    r = 0.05
    s = KouJumpDiffusion(100.0, r, **KOU).simulate(T=1.0, n_steps=2, n_paths=300_000, seed=4).terminal()
    _within(np.exp(-r) * np.maximum(s - K, 0.0), kou_price(100, K, 1.0, r, **KOU))


# ---------------------------------------------------------------- regime switching

Q = np.array([[-0.5, 0.5], [2.0, -2.0]])


def test_regime_occupation_matches_transition_matrix():
    model = RegimeSwitchingGBM(100.0, [0.08, -0.05], [0.15, 0.35], Q)
    res = model.simulate(T=2.0, n_steps=40, n_paths=100_000, seed=5)
    probs = model.transition_matrix(2.0)[0]
    freq = np.mean(res.terminal("regime") == 1.0)
    se = np.sqrt(probs[1] * probs[0] / res.n_paths)
    assert abs(freq - probs[1]) < 4 * se
    assert set(np.unique(res["regime"])) <= {0.0, 1.0}


def test_regime_switching_mean_feynman_kac():
    model = RegimeSwitchingGBM(100.0, [0.08, -0.05], [0.15, 0.35], Q, initial_regime=1)
    s = model.simulate(T=1.0, n_steps=400, n_paths=60_000, seed=6).terminal()
    expected = 100.0 * expm(Q + np.diag([0.08, -0.05]))[1].sum()
    assert model.mean(1.0) == pytest.approx(expected)
    _within(s, expected, k=5.0)


def test_regime_validation():
    with pytest.raises(ValueError):
        RegimeSwitchingGBM(100.0, [0.1, 0.1], [0.2, 0.2], [[-1.0, 0.5], [1.0, -1.0]])


# ---------------------------------------------------------------- commodities

def test_schwartz_one_factor_moments_and_futures():
    model = SchwartzOneFactor(s0=60.0, kappa=1.2, mu=np.log(70.0), sigma=0.35)
    res = model.simulate(grid=TimeGrid([0.0, 0.3, 1.5]), n_paths=200_000, seed=7)
    x = np.log(res.terminal())
    _within(x, model.log_mean(1.5))
    assert x.var() == pytest.approx(model.log_variance(1.5), rel=0.02)
    _within(res.terminal(), model.futures_price(1.5))


def test_schwartz_smith_moments():
    model = SchwartzSmith(chi0=0.1, xi0=np.log(50.0), kappa=1.5, sigma_chi=0.3,
                          mu_xi=0.02, sigma_xi=0.15, rho=0.3)
    res = model.simulate(T=2.0, n_steps=8, n_paths=200_000, seed=8)
    x = np.log(res.terminal())
    np.testing.assert_allclose(x, res.terminal("chi") + res.terminal("xi"))
    _within(x, model.log_mean(2.0))
    assert x.var() == pytest.approx(model.log_variance(2.0), rel=0.02)
    _within(res.terminal(), model.futures_price(2.0))


# ---------------------------------------------------------------- credit

def test_constant_intensity_survival_and_default_times():
    model = ConstantIntensity(0.03)
    res = model.simulate(T=5.0, n_steps=20, n_paths=200_000, seed=9)
    np.testing.assert_allclose(res.terminal("survival"), np.exp(-0.15))
    tau = model.default_times(res, seed=10)
    defaulted = tau <= 5.0
    se = np.sqrt(0.15 * 0.85 / tau.size)
    assert abs(defaulted.mean() - (1 - np.exp(-0.15))) < 4 * se
    # exponential default times: mean of tau given tau <= T
    assert np.all(tau[defaulted] >= 0.0)


def test_cir_intensity_survival_matches_bond_formula():
    model = CIRIntensity(lam0=0.02, kappa=0.8, theta=0.04, sigma=0.12)
    res = model.simulate(T=5.0, n_steps=200, n_paths=60_000, seed=11)
    expected = cir_zcb(0.02, 0.0, 5.0, 0.8, 0.04, 0.12)
    assert model.survival_probability(5.0) == pytest.approx(expected)
    _within(res.terminal("survival"), expected)
    tau = model.default_times(res, seed=12)
    p = 1 - expected
    assert abs(np.mean(tau <= 5.0) - p) < 4 * np.sqrt(p * (1 - p) / tau.size) + 0.002
