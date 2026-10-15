"""Independent checks for GARCH, fractional Brownian motion, Sobol QMC and Euler risk allocations."""

import numpy as np
import pytest
from scipy.stats import norm

import stochPathSim as sps
from stochPathSim.core import sobol_normals
from stochPathSim.risk import normal_portfolio_risk
from stochPathSim.viz import portfolio_risk_chart


def test_garch_exact_one_step_recursion_with_supplied_shocks():
    g = sps.GARCH11(100, mu=0.05, omega=0.001, alpha=0.08, beta=0.90,
                    v0=0.04, q=0.01)
    grid = sps.TimeGrid([0, 0.25, 1.0])
    z = np.array([[[1.0], [-2.0]], [[-1.0], [2.0]]])
    paths = g.simulate(grid=grid, normals=z)
    expected_first = 100*np.exp((0.04-0.02)*0.25 + np.sqrt(0.04*0.25)*z[:, 0, 0])
    first_variance = 0.001 + (0.08*1 + 0.9)*0.04
    second_variance = 0.001 + (0.08*4 + 0.9)*first_variance
    np.testing.assert_allclose(paths["spot"][:, 1], expected_first)
    np.testing.assert_allclose(paths["variance"][:, 1], first_variance)
    np.testing.assert_allclose(paths["variance"][:, 2], second_variance)
    assert g.long_run_variance == pytest.approx(0.05)


def test_garch_stationary_variance_mean():
    g = sps.GARCH11(100, 0.03, 0.002, 0.05, 0.90, v0=0.1)
    res = g.simulate(T=1, n_steps=15, n_paths=60_000, seed=18)
    mean = res.terminal("variance").mean()
    expected = g.expected_variance(15)
    assert abs(mean-expected) < 0.025*expected
    assert np.all(res["variance"] > 0)
    assert np.all(res["spot"] > 0)


def test_gjr_asymmetric_leverage_and_variance_expectation():
    z = np.array([[[-2.0]], [[2.0]]])
    gjr = sps.GJRGARCH11(100, 0.0, 0.001, 0.05, 0.8, 0.2, v0=0.04)
    res = gjr.simulate(T=1, n_steps=1, normals=z)
    assert res["variance"][0, 1] - res["variance"][1, 1] == pytest.approx(0.032)
    assert gjr.long_run_variance == pytest.approx(0.02)
    assert gjr.expected_variance(1) == pytest.approx(0.001 + (0.05+0.8+0.1)*0.04)
    g = sps.GARCH11(100, 0.0, 0.001, 0.05, 0.8, v0=0.04)
    symmetric = sps.GJRGARCH11(100, 0.0, 0.001, 0.05, 0.8, 0.0, v0=0.04)
    z2 = np.random.default_rng(1).normal(size=(7, 5, 1))
    for state in ("spot", "variance"):
        np.testing.assert_array_equal(g.simulate(n_steps=5, normals=z2)[state],
                                      symmetric.simulate(n_steps=5, normals=z2)[state])


@pytest.mark.parametrize("class_name,params", [
    ("GARCH11", dict(omega=-0.001)), ("GARCH11", dict(alpha=0.6, beta=0.4)),
    ("GARCH11", dict(v0=0)), ("GARCH11", dict(mu=np.nan)),
    ("GJRGARCH11", dict(gamma=-0.1)), ("GJRGARCH11", dict(gamma=0.25)),
])
def test_garch_parameter_validation(class_name, params):
    args = dict(s0=100, mu=0.03, omega=0.001, alpha=0.05, beta=0.88, v0=0.04)
    if class_name == "GJRGARCH11":
        args["gamma"] = 0.1
    args.update(params)
    with pytest.raises(ValueError):
        getattr(sps, class_name)(**args)


def test_fractional_brownian_half_matches_standard_brownian_exactly():
    grid = sps.TimeGrid([0, 0.1, 0.3, 0.8, 2.0])
    z = np.random.default_rng(11).standard_normal((12, 4, 1))
    p = sps.FractionalBrownianMotion(x0=2, hurst=0.5, sigma=0.8, drift=0.2)
    out = p.simulate(grid=grid, normals=z)["value"]
    increments = 0.8*np.sqrt(grid.dt)*z[:, :, 0] + 0.2*grid.dt
    expected = 2 + np.concatenate(
        [np.zeros((12, 1)), np.cumsum(increments, axis=1)], axis=1)
    np.testing.assert_allclose(out, expected, atol=1e-14)
    np.testing.assert_allclose(p.increment_covariance(grid), np.diag(grid.dt), atol=1e-15)


@pytest.mark.parametrize("hurst", [0.2, 0.75])
def test_fractional_brownian_covariance_against_theoretical_reference(hurst):
    grid = sps.TimeGrid([0, 0.1, 0.3, 0.9, 1.7])
    sigma = 1.3
    f = sps.FractionalBrownianMotion(hurst=hurst, sigma=sigma)
    paths = f.simulate(grid=grid, n_paths=50_000, seed=123)["value"]
    empirical = np.cov(paths[:, 1:].T, ddof=1)
    t = grid.times[1:]
    reference = 0.5*sigma**2*(
        t[:, None]**(2*hurst) + t[None, :]**(2*hurst)
        - np.abs(t[:, None]-t[None, :])**(2*hurst))
    np.testing.assert_allclose(empirical, reference, atol=0.06, rtol=0.025)
    np.testing.assert_allclose(f.variance(t), sigma**2*t**(2*hurst))
    np.testing.assert_allclose(f.mean(t), np.zeros_like(t))


def test_fractional_brownian_deterministic_limit_and_validation():
    grid = sps.TimeGrid([0, 0.2, 1])
    f = sps.FractionalBrownianMotion(x0=2, hurst=0.1, sigma=0, drift=-0.3)
    np.testing.assert_allclose(f.simulate(grid=grid, n_paths=4)["value"],
                               np.tile(2 - 0.3*grid.times, (4, 1)))
    for args in [dict(hurst=0), dict(hurst=1), dict(sigma=-1), dict(drift=np.inf)]:
        with pytest.raises(ValueError):
            sps.FractionalBrownianMotion(**args)
    with pytest.raises(ValueError):
        f.variance(-1)


def test_sobol_normals_shape_reproducibility_antithetic_and_correlations():
    corr = np.array([[1, 0.65], [0.65, 1]])
    a = sobol_normals(4096, 4, 2, seed=7, correlation=corr)
    b = sobol_normals(4096, 4, 2, seed=7, correlation=corr)
    assert a.shape == (4096, 4, 2)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all()
    assert abs(np.corrcoef(a[:, 0, 0], a[:, 0, 1])[0, 1] - 0.65) < 0.015
    anti = sobol_normals(1024, 3, seed=3, antithetic=True)
    np.testing.assert_array_equal(anti[:512], -anti[512:])


def test_sobol_bridge_preserves_brownian_increment_law_on_irregular_grid():
    grid = sps.TimeGrid([0, 0.05, 0.3, 0.8, 2.0])
    z = sobol_normals(8192, 4, seed=13, grid=grid, brownian_bridge=True)[:, :, 0]
    increments = z * np.sqrt(grid.dt)
    cov = np.cov(increments.T)
    np.testing.assert_allclose(cov, np.diag(grid.dt), atol=0.015, rtol=0.025)
    total = increments.sum(axis=1)
    assert abs(total.mean()) < 0.01
    assert total.var() == pytest.approx(2.0, rel=0.02)
    # The generated normals can directly drive existing simulation APIs.
    res = sps.GBM(100, 0.03, 0.2).simulate(grid=grid, normals=z[:, :, None])
    assert res["spot"].shape == (8192, 5)


@pytest.mark.parametrize("args", [
    (100, 4, 1), (2, 0, 1), (2, 4, 0), (0, 3, 1), (2.5, 3, 1),
])
def test_sobol_rejects_invalid_sizes(args):
    with pytest.raises(ValueError):
        sobol_normals(*args)


def test_sobol_grid_and_correlation_validation():
    with pytest.raises(ValueError):
        sobol_normals(16, 5, grid=sps.TimeGrid([0, 1, 2]))
    with pytest.raises(ValueError):
        sobol_normals(16, 2, n_factors=2, correlation=[[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        sobol_normals(1, 2, antithetic=True)


def test_portfolio_risk_reference_closed_form_and_euler_decomposition():
    w = np.array([2.0, -0.5])
    c = np.array([[0.04, 0.006], [0.006, 0.09]])
    m = np.array([0.01, 0.002])
    p = normal_portfolio_risk(w, c, m, level=0.975, horizon=2)
    sd = np.sqrt(2 * w@c@w)
    mean = 2 * w@m
    z = norm.ppf(0.975)
    assert p.var == pytest.approx(z*sd-mean)
    assert p.es == pytest.approx(norm.pdf(z)/0.025*sd-mean)
    assert p.volatility == pytest.approx(sd)
    assert p.component_var.sum() == pytest.approx(p.var)
    assert p.component_es.sum() == pytest.approx(p.es)
    for j in range(2):
        bump = np.zeros(2)
        bump[j] = 1e-5
        up = normal_portfolio_risk(w+bump, c, m, level=0.975, horizon=2)
        down = normal_portfolio_risk(w-bump, c, m, level=0.975, horizon=2)
        assert p.marginal_var[j] == pytest.approx((up.var-down.var)/(2e-5), rel=1e-7)
        assert p.marginal_es[j] == pytest.approx((up.es-down.es)/(2e-5), rel=1e-7)


def test_portfolio_risk_mc_reference_and_plotly_figure():
    c = np.array([[0.04, -0.01], [-0.01, 0.09]])
    w = np.array([0.7, 0.3])
    p = normal_portfolio_risk(w, c, level=0.99)
    rng = np.random.default_rng(24)
    pnl = rng.multivariate_normal([0, 0], c, size=120_000) @ w
    empirical = np.quantile(-pnl, 0.99)
    assert abs(p.var-empirical) < 0.007
    fig = portfolio_risk_chart(p, ["Equity", "Rates"])
    assert len(fig.data) == 2
    assert list(fig.data[0].x) == ["Equity", "Rates"]
    with pytest.raises(ValueError):
        portfolio_risk_chart(p, ["Equity"])


def test_zero_risk_portfolio_and_input_validation():
    p = normal_portfolio_risk([1, -1], [[0, 0], [0, 0]], mean=[0.03, 0.01])
    assert p.var == pytest.approx(-0.02)
    assert p.es == pytest.approx(-0.02)
    np.testing.assert_allclose(p.component_var, [-0.03, 0.01])
    for w, c in [
        ([1], [[1, 0], [0, 1]]),
        ([1, 2], [[1, 2], [2, 1]]),
        ([1, 2], [[1, 0.2], [0.4, 1]]),
        ([np.nan], [[1]]),
    ]:
        with pytest.raises(ValueError):
            normal_portfolio_risk(w, c)
    with pytest.raises(ValueError):
        normal_portfolio_risk([1], [[1]], level=1)
    with pytest.raises(ValueError):
        normal_portfolio_risk([1], [[1]], horizon=0)
