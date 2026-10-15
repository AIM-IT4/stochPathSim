"""Independent distribution and pricing checks for normal and NIG models."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norminvgauss

import stochPathSim as sps
from stochPathSim.analytics import bachelier_greeks, bachelier_price, nig_price
from stochPathSim.pricing import EuropeanCall, EuropeanPut, price_mc


def within_mc(sample, expected):
    sample = np.asarray(sample)
    se = sample.std(ddof=1) / np.sqrt(sample.size)
    assert abs(sample.mean() - expected) <= 5 * se + 1e-12


def test_arithmetic_exact_steps_with_negative_values():
    grid = sps.TimeGrid([0, 0.1, 0.4, 2])
    z = np.array([[1, -2, 0.5], [-1, 2, -0.5]])
    model = sps.ArithmeticBrownianMotion(-0.02, 0.003, 0.01)
    result = model.simulate(grid=grid, normals=z)
    expected = -0.02 + np.cumsum(0.003 * grid.dt + 0.01 * np.sqrt(grid.dt) * z, axis=1)
    np.testing.assert_allclose(result["spot"][:, 1:], expected)
    np.testing.assert_array_equal(result["spot"][:, 0], [-0.02, -0.02])


def test_arithmetic_moments_and_reproducibility():
    model = sps.ArithmeticBrownianMotion(-2, 0.3, 1.2)
    result = model.simulate(T=2, n_steps=3, n_paths=100_000, seed=17)
    for i, t in enumerate(result.times[1:], 1):
        sample = result["spot"][:, i]
        within_mc(sample, model.mean(t))
        within_mc((sample - model.mean(t))**2, model.variance(t))
    a = model.simulate(n_steps=2, n_paths=10, seed=4)
    b = model.simulate(n_steps=2, n_paths=10, seed=4)
    np.testing.assert_array_equal(a["spot"], b["spot"])


def test_arithmetic_deterministic_and_antithetic():
    grid = sps.TimeGrid([0, 0.2, 0.7, 2])
    deterministic = sps.ArithmeticBrownianMotion(-3, 2, 0).simulate(grid=grid, n_paths=3)
    np.testing.assert_allclose(deterministic["spot"], np.tile(-3 + 2 * grid.times, (3, 1)))
    model = sps.ArithmeticBrownianMotion(-3, 2, 5)
    result = model.simulate(grid=grid, n_paths=100, seed=4, antithetic=True)
    np.testing.assert_allclose(result["spot"].mean(axis=0), model.mean(grid.times), atol=1e-14)


@pytest.mark.parametrize("kind,payoff", [("call", EuropeanCall), ("put", EuropeanPut)])
def test_normal_mc_price_matches_bachelier(kind, payoff):
    model = sps.ArithmeticBrownianMotion(-0.005, 0, 0.015)
    mc = price_mc(model, payoff(-0.01), 2, discount=0.03, n_steps=2,
                  n_paths=100_000, seed=9, antithetic=True)
    exact = bachelier_price(-0.005, -0.01, 2, np.exp(-0.06), 0.015, kind)
    assert abs(mc.price - exact) < 5 * mc.stderr


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("F,K", [(-0.01, -0.005), (0.0, 0.0), (1.5, 1.4)])
def test_normal_greeks_against_price_differences(kind, F, K):
    T, df, sigma = 1.4, 0.96, 0.2
    h = 1e-5
    g = bachelier_greeks(F, K, T, df, sigma, kind)

    def price(f=F, t=T, v=sigma):
        return bachelier_price(f, K, t, df, v, kind)

    assert g["delta"] == pytest.approx((price(f=F+h) - price(f=F-h)) / (2*h), rel=1e-7)
    assert g["gamma"] == pytest.approx((price(f=F+h) - 2*price() + price(f=F-h)) / h**2, rel=1e-5)
    assert g["vega"] == pytest.approx((price(v=sigma+h) - price(v=sigma-h)) / (2*h), rel=1e-7)
    assert g["theta"] == pytest.approx(-(price(t=T+h) - price(t=T-h)) / (2*h), rel=1e-7)


def test_normal_greeks_arrays_and_parity():
    f = np.array([[-0.02], [0.01]])
    k = np.array([-0.01, 0, 0.02])
    call = bachelier_greeks(f, k, 2, 0.95, 0.02)
    put = bachelier_greeks(f, k, 2, 0.95, 0.02, "put")
    assert call["delta"].shape == (2, 3)
    np.testing.assert_allclose(call["delta"] - put["delta"], 0.95)
    for key in ("gamma", "vega", "theta"):
        np.testing.assert_allclose(call[key], put[key])


NIG = dict(alpha=8.0, beta=-3.0, delta=0.25)


@pytest.mark.parametrize("times", [[0, 1.5], [0, 0.05, 0.3, 1.5]])
def test_nig_matches_scipy_distribution_on_arbitrary_grids(times):
    model = sps.NormalInverseGaussian(100, 0.04, q=0.01, **NIG)
    result = model.simulate(grid=sps.TimeGrid(times), n_paths=120_000, seed=23)
    np.testing.assert_array_equal(result["spot"][:, 0], np.full(result.n_paths, 100))
    assert np.all(result["spot"] > 0)
    for i, t in enumerate(times[1:], 1):
        scale = NIG["delta"] * t
        drift = (0.04 - 0.01 - model.compensator) * t
        reference = norminvgauss(NIG["alpha"]*scale, NIG["beta"]*scale, loc=drift, scale=scale)
        log_returns = np.log(result["spot"][:, i] / 100)
        mean, variance = reference.stats(moments="mv")
        assert model.log_return_mean(t) == pytest.approx(mean)
        assert model.log_return_variance(t) == pytest.approx(variance)
        within_mc(log_returns, mean)
        within_mc((log_returns - mean)**2, variance)
        for p in (0.1, 0.5, 0.9):
            freq = np.mean(log_returns <= reference.ppf(p))
            assert abs(freq - p) < 5 * np.sqrt(p*(1-p)/result.n_paths)
        within_mc(result["spot"][:, i], model.mean(t))
        within_mc((result["spot"][:, i] - model.mean(t))**2, model.variance(t))


def test_nig_supplied_normals_and_seed_control_both_random_sources():
    model = sps.NormalInverseGaussian(100, 0.04, **NIG)
    z = np.random.default_rng(12).normal(size=(10, 3, 1))
    a = model.simulate(n_steps=3, normals=z, seed=7)["spot"]
    b = model.simulate(n_steps=3, normals=z, seed=7)["spot"]
    c = model.simulate(n_steps=3, normals=z, seed=8)["spot"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_nig_zero_activity_is_deterministic():
    model = sps.NormalInverseGaussian(100, 0.04, 8, -3, 0, q=0.01)
    result = model.simulate(n_steps=3, n_paths=2, seed=1)
    np.testing.assert_allclose(result["spot"], np.tile(model.mean(result.times), (2, 1)))
    np.testing.assert_array_equal(model.variance(result.times), np.zeros(4))
    np.testing.assert_array_equal(model.log_return_variance(result.times), np.zeros(4))


@pytest.mark.parametrize("kind,payoff", [("call", EuropeanCall), ("put", EuropeanPut)])
@pytest.mark.parametrize("strike", [90, 105, 120])
def test_nig_pricer_matches_mc(kind, payoff, strike):
    model = sps.NormalInverseGaussian(100, 0.04, q=0.01, **NIG)
    mc = price_mc(model, payoff(strike), 1.5, discount=0.04, n_steps=2,
                  n_paths=120_000, seed=8, antithetic=True)
    exact = nig_price(100, strike, 1.5, 0.04, **NIG, option_type=kind, q=0.01)
    assert abs(mc.price - exact) < 5 * mc.stderr


def test_nig_price_against_direct_payoff_integration_and_parity():
    S, K, T, r, q = 100, 105, 1.5, 0.04, 0.01
    # Directly integrate the discounted payoff under the original NIG density,
    # independently of the stock-measure tilting formula used by nig_price.
    a, b, delta = NIG.values()
    correction = delta * (np.sqrt(a*a-b*b) - np.sqrt(a*a-(b+1)**2))
    drift = (r-q-correction)*T
    scale = delta*T
    dist = norminvgauss(a*scale, b*scale, loc=drift, scale=scale)
    reference = np.exp(-r*T) * quad(lambda x: (S*np.exp(x)-K)*dist.pdf(x),
                                    np.log(K/S), 20, epsabs=1e-9)[0]
    call = nig_price(S, K, T, r, **NIG, q=q)
    put = nig_price(S, K, T, r, **NIG, q=q, option_type="put")
    assert call == pytest.approx(reference, abs=1e-8)
    assert call-put == pytest.approx(S*np.exp(-q*T)-K*np.exp(-r*T), abs=1e-9)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_nig_price_expiry_and_deterministic_limit(kind):
    sign = 1 if kind == "call" else -1
    assert nig_price(100, 105, 0, 0.04, **NIG, option_type=kind) == max(sign*(100-105), 0)
    price = nig_price(100, 105, 2, 0.04, 8, -3, 0, option_type=kind, q=0.01)
    assert price == pytest.approx(max(sign*(100*np.exp(-0.02)-105*np.exp(-0.08)), 0))


def test_nig_first_moment_does_not_imply_second_moment():
    model = sps.NormalInverseGaussian(100, 0.03, 1.5, 0, 0.1)
    assert np.isfinite(model.mean(1))
    with pytest.raises(ValueError, match="variance"):
        model.variance(1)
    assert np.isfinite(nig_price(100, 105, 1, 0.03, 1.5, 0, 0.1))


@pytest.mark.parametrize("params", [(-1, 0, -0.1), (np.inf, 0, 1), (1, np.nan, 1)])
def test_invalid_arithmetic_parameters(params):
    with pytest.raises(ValueError):
        sps.ArithmeticBrownianMotion(*params)


@pytest.mark.parametrize("params", [dict(alpha=0), dict(alpha=3), dict(beta=8), dict(delta=-1),
                                     dict(alpha=1.5, beta=1), dict(mu=np.inf), dict(beta=np.nan),
                                     dict(s0=0), dict(q=np.nan)])
def test_invalid_nig_parameters(params):
    args = dict(s0=100, mu=0.04, **NIG)
    args.update(params)
    with pytest.raises(ValueError):
        sps.NormalInverseGaussian(**args)


@pytest.mark.parametrize("T,df,sigma", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, -1), (1, 1, np.inf)])
def test_invalid_greeks_inputs(T, df, sigma):
    with pytest.raises(ValueError):
        bachelier_greeks(1, 1, T, df, sigma)


@pytest.mark.parametrize("K,T", [(0, 1), (100, -1), (np.nan, 1), (100, np.inf)])
def test_invalid_nig_pricing_inputs(K, T):
    with pytest.raises(ValueError):
        nig_price(100, K, T, 0.04, **NIG)


def test_invalid_option_types():
    with pytest.raises(ValueError):
        nig_price(100, 100, 1, 0.04, **NIG, option_type="invalid")
    with pytest.raises(ValueError):
        bachelier_greeks(1, 1, 1, 1, 1, option_type="invalid")
