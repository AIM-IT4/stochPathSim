import numpy as np
import pytest

from stochPathSim.analytics.rates import (
    cir_zcb,
    hull_white_caplet,
    hull_white_zcb,
    hull_white_zcb_option,
    vasicek_zcb,
)
from stochPathSim.core import TimeGrid
from stochPathSim.curves import ZeroCurve
from stochPathSim.processes.rates import (
    CIR,
    BlackDermanToy,
    BlackKarasinski,
    G2pp,
    HoLee,
    HullWhite,
    Vasicek,
)

CURVE = ZeroCurve([0.5, 1.0, 2.0, 5.0, 10.0], [0.02, 0.025, 0.03, 0.035, 0.04])


def assert_mc_close(samples, expected, n_se=4.0):
    samples = np.asarray(samples)
    se = samples.std(ddof=1) / np.sqrt(samples.size)
    assert abs(samples.mean() - expected) < n_se * se + 1e-12, (samples.mean(), expected, se)


def test_vasicek_exact_moments_and_bond_price():
    model = Vasicek(r0=0.03, kappa=0.5, theta=0.05, sigma=0.01)
    res = model.simulate(T=2.0, n_steps=20, n_paths=40_000, seed=11)
    assert res.names == ("rate", "discount")
    assert np.all(res["rate"][:, 0] == 0.03)
    assert np.all(res["discount"][:, 0] == 1.0)
    for t in (0.5, 2.0):
        r_t = res.at(t)
        assert_mc_close(r_t, model.mean(t))
        assert r_t.var(ddof=1) == pytest.approx(model.variance(t), rel=0.03)
    assert_mc_close(res.terminal("discount"), vasicek_zcb(0.03, 0.0, 2.0, 0.5, 0.05, 0.01))


def test_vasicek_exact_on_coarse_grid():
    # one step must give the same distribution as many steps
    model = Vasicek(0.03, 0.5, 0.05, 0.01)
    res = model.simulate(T=3.0, n_steps=1, n_paths=60_000, seed=5)
    assert_mc_close(res.terminal(), model.mean(3.0))
    assert_mc_close(res.terminal("discount"), vasicek_zcb(0.03, 0.0, 3.0, 0.5, 0.05, 0.01))


@pytest.mark.parametrize("scheme", ["exact", "euler"])
def test_cir_moments_bond_price_and_positivity(scheme):
    model = CIR(r0=0.03, kappa=0.8, theta=0.05, sigma=0.1, scheme=scheme)
    res = model.simulate(T=2.0, n_steps=100, n_paths=30_000, seed=3)
    assert np.all(res["rate"] >= 0.0)
    assert_mc_close(res.terminal(), model.mean(2.0))
    assert res.terminal().var(ddof=1) == pytest.approx(model.variance(2.0), rel=0.05)
    expected = cir_zcb(0.03, 0.0, 2.0, 0.8, 0.05, 0.1)
    assert abs(res.terminal("discount").mean() - expected) < 2e-4


def test_cir_exact_stays_non_negative_when_feller_fails():
    model = CIR(r0=0.01, kappa=0.5, theta=0.02, sigma=0.3)
    assert not model.feller_satisfied
    assert CIR(0.03, 0.8, 0.05, 0.1).feller_satisfied
    res = model.simulate(T=1.0, n_steps=50, n_paths=5_000, seed=1)
    assert np.all(res["rate"] >= 0.0)
    assert_mc_close(res.terminal(), model.mean(1.0))


@pytest.mark.parametrize(
    "model",
    [
        HullWhite(a=0.1, sigma=0.01, curve=CURVE),
        HoLee(sigma=0.01, curve=CURVE),
        G2pp(a=0.5, b=0.1, sigma=0.01, eta=0.008, rho=-0.6, curve=CURVE),
    ],
    ids=["hull-white", "ho-lee", "g2pp"],
)
def test_curve_fitted_models_reproduce_the_curve(model):
    res = model.simulate(T=5.0, n_steps=50, n_paths=20_000, seed=1, antithetic=True)
    assert res["rate"][0, 0] == pytest.approx(model.r0)
    assert model.r0 == pytest.approx(float(CURVE.forward_rate(0.0)), abs=1e-10)
    for t in (1.0, 2.5, 5.0):
        assert_mc_close(res.at(t, "discount"), float(CURVE.discount(t)))


def test_curve_fitted_models_are_grid_independent():
    model = HullWhite(0.1, 0.01, CURVE)
    coarse = model.simulate(T=5.0, n_steps=1, n_paths=40_000, seed=8)
    assert_mc_close(coarse.terminal("discount"), float(CURVE.discount(5.0)))


def test_hull_white_flat_rate_curve_argument():
    model = HullWhite(0.05, 0.01, 0.03)
    assert model.r0 == pytest.approx(0.03, abs=1e-8)


def test_hull_white_bond_option_monte_carlo():
    a, sigma, T, S, K = 0.1, 0.01, 1.0, 5.0, 0.86
    model = HullWhite(a, sigma, CURVE)
    res = model.simulate(T=T, n_steps=4, n_paths=50_000, seed=21)
    p_ts = hull_white_zcb(res.terminal(), T, S, a, sigma, CURVE)
    disc = res.terminal("discount")
    assert_mc_close(disc * p_ts, float(CURVE.discount(S)))
    for kind, payoff in (("call", np.maximum(p_ts - K, 0.0)), ("put", np.maximum(K - p_ts, 0.0))):
        assert_mc_close(disc * payoff, float(hull_white_zcb_option(T, S, K, a, sigma, CURVE, kind)))


def test_hull_white_caplet_monte_carlo():
    a, sigma, T1, T2, K = 0.1, 0.012, 2.0, 2.5, 0.03
    model = HullWhite(a, sigma, CURVE)
    res = model.simulate(T=T1, n_steps=8, n_paths=50_000, seed=4)
    p = hull_white_zcb(res.terminal(), T1, T2, a, sigma, CURVE)
    tau = T2 - T1
    libor = (1.0 / p - 1.0) / tau
    pv = res.terminal("discount") * tau * np.maximum(libor - K, 0.0) * p
    assert_mc_close(pv, float(hull_white_caplet(T1, T2, K, a, sigma, CURVE)))


def test_g2pp_states_and_correlation():
    model = G2pp(a=0.5, b=0.1, sigma=0.01, eta=0.008, rho=-0.6, curve=CURVE)
    res = model.simulate(T=1.0, n_steps=10, n_paths=40_000, seed=2)
    assert res.names == ("rate", "x", "y", "discount")
    x, y = res.terminal("x"), res.terminal("y")
    assert np.corrcoef(x, y)[0, 1] < -0.5
    assert np.allclose(res["rate"], res["x"] + res["y"] + model.shift(res.times))
    vx = 0.01**2 / (2 * 0.5) * (1 - np.exp(-2 * 0.5))
    assert x.var(ddof=1) == pytest.approx(vx, rel=0.03)


def test_black_karasinski_log_rate_moments_and_positivity():
    model = BlackKarasinski(r0=0.03, a=0.3, sigma=0.2, theta=0.3 * np.log(0.04))
    res = model.simulate(T=3.0, n_steps=30, n_paths=40_000, seed=6)
    assert np.all(res["rate"] > 0.0)
    log_r = np.log(res.terminal())
    assert_mc_close(log_r, model.log_rate_mean(3.0))
    assert log_r.var(ddof=1) == pytest.approx(model.log_rate_variance(3.0), rel=0.03)


def test_black_karasinski_callable_theta_matches_constant():
    const = BlackKarasinski(0.03, 0.3, 0.2, theta=-1.0)
    func = BlackKarasinski(0.03, 0.3, 0.2, theta=lambda t: -1.0)
    a = const.simulate(T=1.0, n_steps=12, n_paths=100, seed=1)
    b = func.simulate(T=1.0, n_steps=12, n_paths=100, seed=1)
    assert np.allclose(a["rate"], b["rate"])


def test_black_derman_toy_lognormal():
    model = BlackDermanToy(r0=0.03, sigma=0.2, theta=0.01)
    res = model.simulate(T=2.0, n_steps=20, n_paths=40_000, seed=9)
    log_r = np.log(res.terminal())
    assert_mc_close(log_r, np.log(0.03) + 0.01 * 2.0)
    assert log_r.var(ddof=1) == pytest.approx(0.2**2 * 2.0, rel=0.03)
    vec = BlackDermanToy(0.03, 0.2, theta=lambda t: 0.01 + 0.0 * np.asarray(t))
    assert np.allclose(vec.simulate(T=2.0, n_steps=20, n_paths=10, seed=9)["rate"],
                       model.simulate(T=2.0, n_steps=20, n_paths=10, seed=9)["rate"])


def test_discount_is_consistent_with_rate_for_trapezoid_models():
    res = CIR(0.03, 0.8, 0.05, 0.1, scheme="euler").simulate(T=1.0, n_steps=10, n_paths=5, seed=1)
    r, dt = res["rate"], np.diff(res.times)
    expected = np.exp(-np.sum(0.5 * (r[:, :-1] + r[:, 1:]) * dt, axis=1))
    assert np.allclose(res.terminal("discount"), expected)


def test_non_uniform_grid():
    grid = TimeGrid([0.0, 0.1, 0.5, 2.0])
    res = HullWhite(0.1, 0.01, CURVE).simulate(grid=grid, n_paths=30_000, seed=3)
    assert res.times.tolist() == [0.0, 0.1, 0.5, 2.0]
    assert_mc_close(res.terminal("discount"), float(CURVE.discount(2.0)))


def test_antithetic_vasicek_rate_mean_is_exact():
    model = Vasicek(0.03, 0.5, 0.05, 0.01)
    res = model.simulate(T=1.0, n_steps=5, n_paths=1000, seed=1, antithetic=True)
    assert res.terminal().mean() == pytest.approx(model.mean(1.0), abs=1e-12)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Vasicek(0.03, 0.0, 0.05, 0.01),
        lambda: Vasicek(0.03, 0.5, 0.05, -0.01),
        lambda: CIR(-0.01, 0.5, 0.05, 0.1),
        lambda: CIR(0.03, 0.5, 0.05, 0.1, scheme="milstein"),
        lambda: HullWhite(0.0, 0.01, CURVE),
        lambda: HoLee(-0.01, CURVE),
        lambda: G2pp(0.5, 0.1, 0.01, 0.01, 1.5, CURVE),
        lambda: BlackKarasinski(0.0, 0.1, 0.2),
        lambda: BlackDermanToy(-0.01, 0.2),
    ],
)
def test_invalid_parameters_raise(factory):
    with pytest.raises(ValueError):
        factory()


def test_normals_with_wrong_factor_count_raise():
    with pytest.raises(ValueError):
        HullWhite(0.1, 0.01, CURVE).simulate(T=1.0, n_steps=4, normals=np.zeros((10, 4, 1)))
