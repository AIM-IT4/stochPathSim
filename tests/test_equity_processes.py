import numpy as np
import pytest

from stochPathSim.analytics import black76_price, bs_price, heston_price, merton_price, sabr_implied_vol
from stochPathSim.processes.equity import (
    CEV,
    SABR,
    Bates,
    Heston,
    MertonJumpDiffusion,
    MultiAssetGBM,
    VarianceGamma,
)
from stochPathSim.processes.gbm import GBM
from stochPathSim.processes.ou import OrnsteinUhlenbeck


def discounted_call(terminal, K, r, T):
    """Discounted call payoff mean and its standard error."""
    payoff = np.exp(-r * T) * np.maximum(terminal - K, 0.0)
    return payoff.mean(), payoff.std(ddof=1) / np.sqrt(payoff.size)


def assert_within(estimate, stderr, expected, n_sigma=4.0):
    assert abs(estimate - expected) < n_sigma * stderr, (estimate, expected, stderr)


def test_gbm_call_matches_black_scholes():
    res = GBM(100.0, 0.05, 0.2).simulate(T=1.0, n_steps=1, n_paths=100_000, seed=11)
    price, se = discounted_call(res.terminal(), 100.0, 0.05, 1.0)
    assert_within(price, se, bs_price(100.0, 100.0, 1.0, 0.05, 0.2))


def test_merton_call_matches_series():
    proc = MertonJumpDiffusion(100.0, 0.05, 0.2, lam=0.5, jump_mean=-0.1, jump_std=0.15)
    res = proc.simulate(T=1.0, n_steps=4, n_paths=100_000, seed=12)
    price, se = discounted_call(res.terminal(), 100.0, 0.05, 1.0)
    assert_within(price, se, merton_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.5, -0.1, 0.15))
    mean, se_mean = res.terminal().mean(), res.terminal().std(ddof=1) / np.sqrt(res.n_paths)
    assert_within(mean, se_mean, proc.mean(1.0))


@pytest.mark.parametrize("scheme, n_steps", [("qe", 8), ("euler", 100)])
def test_heston_call_matches_semi_analytic(scheme, n_steps):
    proc = Heston(100.0, 0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, scheme=scheme)
    res = proc.simulate(T=1.0, n_steps=n_steps, n_paths=60_000, seed=13)
    price, se = discounted_call(res.terminal(), 100.0, 0.05, 1.0)
    # allow a small discretisation bias on top of the statistical error
    assert abs(price - heston_price(100, 100, 1.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7)) < 4 * se + 0.03
    assert np.all(res["variance"] >= 0.0)


def test_heston_qe_is_martingale_with_dividends():
    proc = Heston(100.0, 0.03, v0=0.05, kappa=1.5, theta=0.06, xi=0.5, rho=-0.5, q=0.01)
    res = proc.simulate(T=1.0, n_steps=12, n_paths=60_000, seed=14, antithetic=True)
    s = res.terminal()
    assert_within(s.mean(), s.std(ddof=1) / np.sqrt(s.size), proc.mean(1.0))
    v = res["variance"][:, -1]
    assert_within(v.mean(), v.std(ddof=1) / np.sqrt(v.size), proc.variance_mean(1.0))


def test_bates_mean_and_no_jump_limit():
    proc = Bates(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, lam=0.5, jump_mean=-0.1, jump_std=0.15)
    s = proc.simulate(T=1.0, n_steps=12, n_paths=60_000, seed=15).terminal()
    assert_within(s.mean(), s.std(ddof=1) / np.sqrt(s.size), proc.mean(1.0))

    no_jumps = Bates(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, lam=0.0, jump_mean=-0.1, jump_std=0.15)
    heston = Heston(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
    a = no_jumps.simulate(T=1.0, n_steps=4, n_paths=1000, seed=16).terminal()
    b = heston.simulate(T=1.0, n_steps=4, n_paths=1000, seed=16).terminal()
    np.testing.assert_allclose(a, b)


def test_sabr_call_matches_hagan():
    F, K, T, alpha, beta, rho, nu = 0.03, 0.03, 2.0, 0.03, 0.5, -0.3, 0.4
    res = SABR(F, alpha, beta, rho, nu).simulate(T=T, n_steps=200, n_paths=40_000, seed=17)
    payoff = np.maximum(res.terminal() - K, 0.0)
    price, se = payoff.mean(), payoff.std(ddof=1) / np.sqrt(payoff.size)
    hagan = black76_price(F, K, T, 1.0, sabr_implied_vol(F, K, T, alpha, beta, rho, nu))
    # Hagan's expansion is itself an approximation; allow 2% on top of the MC error
    assert abs(price - hagan) < 4 * se + 0.02 * hagan
    assert np.all(res["forward"] >= 0.0) and np.all(res["vol"] > 0.0)


def test_cev_beta_one_is_gbm_and_mean():
    proc = CEV(100.0, 0.05, 2.0, beta=0.5)
    s = proc.simulate(T=1.0, n_steps=100, n_paths=40_000, seed=18).terminal()
    assert np.all(s >= 0.0)
    assert_within(s.mean(), s.std(ddof=1) / np.sqrt(s.size), 100.0 * np.exp(0.05), n_sigma=5)


def test_multi_asset_gbm_correlation_and_means():
    corr = np.array([[1.0, 0.6, -0.2], [0.6, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    proc = MultiAssetGBM([100.0, 50.0, 20.0], [0.05, 0.03, 0.0], [0.2, 0.3, 0.4], corr)
    res = proc.simulate(T=1.0, n_steps=10, n_paths=40_000, seed=19)
    assert res.names == ("asset_0", "asset_1", "asset_2")
    rets = np.stack([np.diff(np.log(res[n]), axis=1).ravel() for n in res.names])
    np.testing.assert_allclose(np.corrcoef(rets), corr, atol=0.01)
    for i, name in enumerate(res.names):
        s = res.terminal(name)
        assert_within(s.mean(), s.std(ddof=1) / np.sqrt(s.size), proc.mean(1.0)[i])


def test_multi_asset_gbm_rejects_invalid_correlation():
    with pytest.raises(ValueError):
        MultiAssetGBM([1.0, 1.0], 0.0, 0.2, [[1.0, 1.5], [1.5, 1.0]])


def test_variance_gamma_mean():
    proc = VarianceGamma(100.0, 0.05, sigma=0.2, nu=0.3, theta=-0.1)
    s = proc.simulate(T=1.0, n_steps=10, n_paths=60_000, seed=20).terminal()
    assert_within(s.mean(), s.std(ddof=1) / np.sqrt(s.size), proc.mean(1.0))


def test_variance_gamma_parameter_check():
    with pytest.raises(ValueError):
        VarianceGamma(100.0, 0.0, sigma=1.0, nu=5.0, theta=0.5)


def test_ou_exact_moments():
    proc = OrnsteinUhlenbeck(1.0, kappa=2.0, theta=0.5, sigma=0.3)
    x = proc.simulate(T=1.0, n_steps=5, n_paths=60_000, seed=21).terminal()
    assert_within(x.mean(), x.std(ddof=1) / np.sqrt(x.size), proc.mean(1.0))
    assert x.var(ddof=1) == pytest.approx(proc.variance(1.0), rel=0.03)


def test_processes_start_at_initial_state_and_are_reproducible():
    proc = Heston(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
    a = proc.simulate(T=0.5, n_steps=6, n_paths=50, seed=3)
    b = proc.simulate(T=0.5, n_steps=6, n_paths=50, seed=3)
    np.testing.assert_array_equal(a["spot"], b["spot"])
    assert np.all(a["spot"][:, 0] == 100.0) and np.all(a["variance"][:, 0] == 0.04)
    assert a["spot"].shape == (50, 7)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        Heston(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, rho=-1.5)
    with pytest.raises(ValueError):
        Heston(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, scheme="milstein")
    with pytest.raises(ValueError):
        SABR(0.03, 0.03, beta=1.5, rho=0.0, nu=0.4)
    with pytest.raises(ValueError):
        OrnsteinUhlenbeck(0.0, kappa=0.0, theta=0.0, sigma=0.1)
