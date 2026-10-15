"""Independent checks for displaced diffusion, option Greeks, and CIR++."""

import numpy as np
import pytest
from scipy.stats import norm

import stochPathSim as sps
from stochPathSim.analytics import (
    cir_zcb,
    displaced_diffusion_greeks,
    displaced_diffusion_price,
)
from stochPathSim.pricing import EuropeanCall, EuropeanPut, price_mc


@pytest.mark.parametrize("kind,payoff", [("call", EuropeanCall), ("put", EuropeanPut)])
@pytest.mark.parametrize("strike", [-0.025, -0.012, 0.0])
def test_displaced_forward_prices_against_mc(kind, payoff, strike):
    forward, shift, vol, maturity, rate = -0.015, 0.06, 0.45, 1.5, 0.03
    model = sps.DisplacedDiffusion(forward, mu=0, sigma=vol, shift=shift)
    mc = price_mc(model, payoff(strike), maturity, discount=rate,
                  n_steps=1, n_paths=100_000, seed=31, antithetic=True)
    closed = displaced_diffusion_price(forward, strike, maturity, np.exp(-rate*maturity),
                                       vol, shift, option_type=kind)
    assert abs(mc.price - closed) < 5 * mc.stderr + 1e-12


def test_displaced_matches_independent_shifted_black_and_put_call_parity():
    f, k, t, df, vol, shift = -0.01, -0.018, 2.0, 0.97, 0.4, 0.05
    sf, sk = f + shift, k + shift
    d1 = (np.log(sf/sk) + 0.5*vol**2*t) / (vol*np.sqrt(t))
    d2 = d1 - vol*np.sqrt(t)
    reference = df * (sf*norm.cdf(d1) - sk*norm.cdf(d2))
    call = displaced_diffusion_price(f, k, t, df, vol, shift)
    put = displaced_diffusion_price(f, k, t, df, vol, shift, "put")
    assert call == pytest.approx(reference, abs=1e-12)
    assert call - put == pytest.approx(df*(f-k), abs=1e-12)


def test_displaced_moments_custom_grid_and_antithetic():
    model = sps.DisplacedDiffusion(-0.015, 0.07, 0.3, 0.06, q=0.02)
    grid = sps.TimeGrid([0, 0.1, 0.4, 1.5])
    paths = model.simulate(grid=grid, n_paths=60_000, seed=21)
    assert np.all(paths["spot"] > -0.06)
    np.testing.assert_allclose(paths["spot"][:, 0], -0.015)
    assert paths["spot"].shape == (60_000, 4)
    for i, t in enumerate(grid.times[1:], 1):
        x = paths["spot"][:, i]
        assert abs(x.mean() - model.mean(t)) < 5 * x.std(ddof=1) / np.sqrt(x.size)
        v = model.variance(t)
        assert abs(x.var(ddof=1) - v) < 0.06 * v

    z = np.random.default_rng(2).normal(size=(6, 3, 1))
    a = model.simulate(grid=grid, normals=z)["spot"]
    b = model.simulate(grid=grid, normals=z)["spot"]
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a[:, 0], -0.015)


def test_displaced_closed_form_deterministic_and_strike_below_lower_bound():
    f, shift, df = -0.025, 0.04, 0.98
    for k in [-0.1, -0.04]:
        assert displaced_diffusion_price(f, k, 2, df, 0.3, shift) == pytest.approx(df*(f-k))
        assert displaced_diffusion_price(f, k, 2, df, 0.3, shift, "put") == 0.0
    assert displaced_diffusion_price(f, -0.02, 1, df, 0, shift) == 0.0
    np.testing.assert_allclose(
        sps.DisplacedDiffusion(f, 0, 0, shift).simulate(n_steps=2, n_paths=4, seed=1)["spot"],
        f, atol=1e-15)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_displaced_greeks_finite_differences(kind):
    f, k, t, df, vol, shift = -0.01, -0.005, 1.3, 0.97, 0.3, 0.08
    h = 1e-5
    def price(x=f, T=t, v=vol):
        return displaced_diffusion_price(x, k, T, df, v, shift, kind)
    g = displaced_diffusion_greeks(f, k, t, df, vol, shift, kind)
    assert g["delta"] == pytest.approx((price(x=f+h)-price(x=f-h))/(2*h), rel=1e-7)
    assert g["gamma"] == pytest.approx((price(x=f+h)-2*price()+price(x=f-h))/h**2, rel=3e-4)
    assert g["vega"] == pytest.approx((price(v=vol+h)-price(v=vol-h))/(2*h), rel=1e-7)
    assert g["theta"] == pytest.approx(-(price(T=t+h)-price(T=t-h))/(2*h), rel=1e-7)


def test_displaced_greeks_broadcast_and_linear_payoff():
    f = np.array([[-0.03], [0.01]])
    k = np.array([-0.1, -0.02, 0.01])
    g = displaced_diffusion_greeks(f, k, 1.0, 0.99, 0.2, 0.06)
    p = displaced_diffusion_greeks(f, k, 1.0, 0.99, 0.2, 0.06, "put")
    assert g["delta"].shape == (2, 3)
    np.testing.assert_allclose(g["delta"] - p["delta"], 0.99)
    for key in ("gamma", "vega", "theta"):
        np.testing.assert_allclose(g[key], p[key])
        assert np.all(g[key][:, 0] == 0)


@pytest.mark.parametrize("parameters", [
    dict(s0=-0.1, shift=0.05), dict(sigma=-0.1), dict(shift=np.nan),
    dict(mu=np.inf), dict(q=np.nan),
])
def test_displaced_invalid_process(parameters):
    args = dict(s0=-0.01, mu=0, sigma=0.2, shift=0.05)
    args.update(parameters)
    with pytest.raises(ValueError):
        sps.DisplacedDiffusion(**args)


@pytest.mark.parametrize("parameters", [
    dict(F=-0.1), dict(sigma=-0.1), dict(T=-1), dict(df=0),
    dict(shift=np.nan), dict(option_type="other"),
])
def test_displaced_invalid_pricer(parameters):
    args = dict(F=-0.01, K=0, T=1, df=0.98, sigma=0.2, shift=0.05)
    args.update(parameters)
    with pytest.raises(ValueError):
        displaced_diffusion_price(**args)


def test_cirpp_exact_factor_transition_and_discount_decomposition():
    curve = sps.ZeroCurve.flat(0.025)
    model = sps.CIRPlusPlus(0.04, 1.5, 0.03, 0.12, curve)
    base = sps.CIR(0.04, 1.5, 0.03, 0.12)
    grid = sps.TimeGrid([0, 0.1, 0.3, 1])
    pp = model.simulate(grid=grid, n_paths=1000, seed=44)
    raw = base.simulate(grid=grid, n_paths=1000, seed=44)
    np.testing.assert_array_equal(pp["factor"], raw["rate"])
    expected = raw["discount"] * (curve.discount(grid.times) /
                                  cir_zcb(0.04, 0, grid.times, 1.5, 0.03, 0.12))
    np.testing.assert_allclose(pp["discount"], expected, rtol=1e-13)
    np.testing.assert_allclose(pp["rate"], raw["rate"] + model.shift(grid.times))
    assert model.r0 == pytest.approx(curve.forward_rate(0), abs=1e-10)
    assert pp["discount"][:, 0].tolist() == [1.0]*pp.n_paths
    assert np.all(pp["factor"] >= 0)


def test_cirpp_shift_uses_analytic_cir_forward_not_numerical_approximation():
    kappa, theta, sigma, x0 = 1.7, 0.025, 0.12, 0.04
    model = sps.CIRPlusPlus(x0, kappa, theta, sigma, curve=0.035)
    assert model.cir_forward_rate(0) == pytest.approx(x0, abs=1e-14)
    for t in [0.3, 1.2, 4.0]:
        h = 1e-5
        fd = -(np.log(cir_zcb(x0, 0, t+h, kappa, theta, sigma))
               - np.log(cir_zcb(x0, 0, t-h, kappa, theta, sigma))) / (2*h)
        assert model.cir_forward_rate(t) == pytest.approx(fd, abs=1e-9)


def test_cirpp_curve_fit_mc_and_factor_moments():
    curve = sps.ZeroCurve([0.25, 1, 3], [0.02, 0.026, 0.035])
    model = sps.CIRPlusPlus(0.03, 1.2, 0.035, 0.14, curve)
    paths = model.simulate(T=2.0, n_steps=96, n_paths=25_000, seed=57)
    assert np.all(paths["factor"] >= 0.0)
    x = paths.terminal("factor")
    assert abs(x.mean() - model._cir.mean(2)) < 5*x.std(ddof=1)/np.sqrt(x.size)
    assert abs(paths["discount"][:, -1].mean() - curve.discount(2)) < 0.0015
    assert model.mean(2) == pytest.approx(model.shift(2) + model._cir.mean(2))
    assert model.variance(2) == pytest.approx(model._cir.variance(2))


def test_cirpp_euler_reproducibility_and_negative_shifted_rates():
    model = sps.CIRPlusPlus(0.05, 1.5, 0.02, 0.1, curve=-0.01, scheme="euler")
    z = np.random.default_rng(8).normal(size=(10, 4, 1))
    a = model.simulate(normals=z, n_steps=4)
    b = model.simulate(normals=z, n_steps=4)
    for name in ("rate", "factor", "discount"):
        np.testing.assert_array_equal(a[name], b[name])
    assert np.any(a["rate"] < 0)
    assert np.all(a["discount"] > 0)


@pytest.mark.parametrize("params", [
    dict(x0=-0.02), dict(kappa=0), dict(theta=0), dict(sigma=-0.1),
    dict(sigma=0), dict(kappa=np.nan), dict(x0=np.inf), dict(scheme="invalid"),
])
def test_cirpp_invalid_params(params):
    args = dict(x0=0.02, kappa=1.5, theta=0.03, sigma=0.12, curve=0.02)
    args.update(params)
    with pytest.raises(ValueError):
        sps.CIRPlusPlus(**args)
