import numpy as np
import pytest
from scipy.stats import norm

from stochPathSim.core import SimulationResult, TimeGrid
from stochPathSim.curves import ZeroCurve
from stochPathSim.pricing import (
    AsianArithmetic,
    AsianGeometric,
    Barrier,
    Basket,
    Digital,
    EuropeanCall,
    EuropeanPut,
    LookbackFixed,
    LookbackFloating,
    Spread,
    crr_price,
    longstaff_schwartz,
    price_mc,
)
from stochPathSim.processes import GBM

S0, K, R, Q, SIGMA, T = 100.0, 100.0, 0.05, 0.01, 0.2, 1.0


def bs(S, K, T, r, sigma, option_type="call", q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def geometric_asian(S, K, r, q, sigma, fixing_times, option_type="call"):
    """Discretely monitored geometric Asian (Kemna-Vorst type closed form)."""
    t = np.asarray(fixing_times)
    T = t[-1]
    mu = np.log(S) + (r - q - 0.5 * sigma**2) * t.mean()
    var = sigma**2 * np.minimum.outer(t, t).mean()
    sd = np.sqrt(var)
    d2 = (mu - np.log(K)) / sd
    d1 = d2 + sd
    fwd = np.exp(mu + 0.5 * var)
    if option_type == "call":
        return np.exp(-r * T) * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
    return np.exp(-r * T) * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))


def down_and_out_call(S, K, H, T, r, sigma, q=0.0):
    """Continuously monitored down-and-out call with H <= K (Reiner-Rubinstein)."""
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    y = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
    c_di = S * np.exp(-q * T) * (H / S) ** (2 * lam) * norm.cdf(y) - K * np.exp(-r * T) * (H / S) ** (
        2 * lam - 2
    ) * norm.cdf(y - sigma * np.sqrt(T))
    return bs(S, K, T, r, sigma, "call", q) - c_di


@pytest.fixture
def gbm():
    return GBM(S0, R, SIGMA, q=Q)


@pytest.mark.parametrize("payoff,option_type", [(EuropeanCall(K), "call"), (EuropeanPut(K), "put")])
def test_european_matches_black_scholes(gbm, payoff, option_type):
    res = price_mc(gbm, payoff, T, discount=R, n_steps=1, n_paths=200_000, seed=1)
    exact = bs(S0, K, T, R, SIGMA, option_type, Q)
    assert abs(res.price - exact) < 4 * res.stderr
    lo, hi = res.conf_int(0.999)
    assert lo < exact < hi


def test_antithetic_reduces_stderr(gbm):
    plain = price_mc(gbm, EuropeanCall(K), T, discount=R, n_steps=1, n_paths=100_000, seed=2)
    anti = price_mc(gbm, EuropeanCall(K), T, discount=R, n_steps=1, n_paths=100_000, seed=2, antithetic=True)
    assert anti.stderr < plain.stderr
    assert abs(anti.price - bs(S0, K, T, R, SIGMA, "call", Q)) < 4 * anti.stderr


def test_geometric_asian_matches_closed_form(gbm):
    n = 12
    fixings = np.linspace(T / n, T, n)
    res = price_mc(gbm, AsianGeometric(K), T, discount=R, n_steps=n, n_paths=200_000, seed=3)
    exact = geometric_asian(S0, K, R, Q, SIGMA, fixings)
    assert abs(res.price - exact) < 4 * res.stderr


def test_control_variate_reduces_stderr(gbm):
    n = 12
    fixings = np.linspace(T / n, T, n)
    exact_geo = geometric_asian(S0, K, R, Q, SIGMA, fixings)
    kwargs = dict(discount=R, n_steps=n, n_paths=50_000, seed=4)
    plain = price_mc(gbm, AsianArithmetic(K), T, **kwargs)
    cv = price_mc(gbm, AsianArithmetic(K), T, control_variate=(AsianGeometric(K), exact_geo), **kwargs)
    assert cv.stderr < plain.stderr / 10
    assert abs(cv.price - plain.price) < 4 * plain.stderr
    # arithmetic average dominates geometric
    assert cv.price > exact_geo


def test_asian_fixing_times_must_be_on_grid(gbm):
    grid = TimeGrid.with_times(T, 4, [0.3])
    res = gbm.simulate(grid=grid, n_paths=10, seed=0)
    AsianArithmetic(K, fixing_times=[0.3, 1.0])(res)
    with pytest.raises(ValueError):
        AsianArithmetic(K, fixing_times=[0.31])(res)


@pytest.mark.parametrize("direction,H", [("up", 120.0), ("down", 85.0)])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_barrier_in_out_parity(gbm, direction, H, option_type):
    res = gbm.simulate(T, 50, 5_000, seed=5)
    vanilla = (EuropeanCall(K) if option_type == "call" else EuropeanPut(K))(res)
    out = Barrier(K, H, f"{direction}-and-out", option_type)(res)
    knock_in = Barrier(K, H, f"{direction}-and-in", option_type)(res)
    np.testing.assert_allclose(out + knock_in, vanilla)


def test_barrier_rebate_and_validation(gbm):
    res = gbm.simulate(T, 50, 2_000, seed=6)
    b = Barrier(K, 95.0, "down-and-out", "call", rebate=3.0)
    pay = b(res)
    assert np.all(pay[b.knocked(res.get(), res.times)] == 3.0)
    with pytest.raises(ValueError):
        Barrier(K, 95.0, "sideways-and-out")


def test_down_and_out_matches_continuous_with_correction():
    gbm = GBM(S0, R, 0.3)
    H, n = 90.0, 50
    exact = down_and_out_call(S0, K, H, T, R, 0.3)
    corrected = price_mc(
        gbm,
        Barrier(K, H, "down-and-out", "call", sigma=0.3),
        T,
        discount=R,
        n_steps=n,
        n_paths=200_000,
        seed=7,
    )
    raw = price_mc(
        gbm, Barrier(K, H, "down-and-out", "call"), T, discount=R, n_steps=n, n_paths=200_000, seed=7
    )
    assert abs(corrected.price - exact) < 4 * corrected.stderr + 0.03
    # discrete monitoring without correction overprices a knock-out
    assert raw.price > corrected.price


def test_digital_matches_closed_form(gbm):
    res = price_mc(gbm, Digital(K, "call", cash=10.0), T, discount=R, n_steps=1, n_paths=200_000, seed=8)
    d2 = (np.log(S0 / K) + (R - Q - 0.5 * SIGMA**2) * T) / (SIGMA * np.sqrt(T))
    exact = 10.0 * np.exp(-R * T) * norm.cdf(d2)
    assert abs(res.price - exact) < 4 * res.stderr


def test_lookbacks_on_hand_built_paths():
    times = np.array([0.0, 0.5, 1.0])
    paths = np.array([[100.0, 120.0, 110.0], [100.0, 80.0, 90.0]])
    np.testing.assert_allclose(LookbackFixed(105.0, "call")(paths, times), [15.0, 0.0])
    np.testing.assert_allclose(LookbackFixed(105.0, "put")(paths, times), [5.0, 25.0])
    np.testing.assert_allclose(LookbackFloating("call")(paths, times), [10.0, 10.0])
    np.testing.assert_allclose(LookbackFloating("put")(paths, times), [10.0, 10.0])


def test_lookback_dominates_european(gbm):
    res = gbm.simulate(T, 50, 5_000, seed=9)
    assert np.all(LookbackFixed(K, "call")(res) >= EuropeanCall(K)(res))


def test_basket_and_spread():
    times = np.array([0.0, 1.0])
    result = SimulationResult(
        times, {"a": np.array([[1.0, 110.0], [1.0, 90.0]]), "b": np.array([[1.0, 100.0], [1.0, 100.0]])}
    )
    np.testing.assert_allclose(Spread(0.0)(result), [10.0, 0.0])
    np.testing.assert_allclose(Spread(0.0, "put", states=("a", "b"))(result), [0.0, 10.0])
    np.testing.assert_allclose(Basket([0.5, 0.5], 100.0)(result), [5.0, 0.0])
    with pytest.raises(ValueError):
        Basket([1.0], 100.0)(result)
    with pytest.raises(TypeError):
        Basket([1.0, 1.0], 100.0)(np.ones((2, 2)))


def test_zero_curve_discounting(gbm):
    curve = ZeroCurve.flat(R)
    a = price_mc(gbm, EuropeanCall(K), T, discount=curve, n_steps=1, n_paths=10_000, seed=10)
    b = price_mc(gbm, EuropeanCall(K), T, discount=R, n_steps=1, n_paths=10_000, seed=10)
    assert a.price == pytest.approx(b.price, rel=1e-12)


def test_pathwise_discount_state_is_used():
    class WithDiscount(GBM):
        state_names = ("spot", "discount")

        def _simulate(self, grid, normals, rng):
            paths = super()._simulate(grid, normals, rng)
            paths["discount"] = np.broadcast_to(np.exp(-R * grid.times), paths["spot"].shape).copy()
            return paths

    proc = WithDiscount(S0, R, SIGMA)
    a = price_mc(proc, EuropeanCall(K), T, n_steps=1, n_paths=10_000, seed=11)
    b = price_mc(proc, EuropeanCall(K), T, discount=R, n_steps=1, n_paths=10_000, seed=11)
    assert a.price == pytest.approx(b.price, rel=1e-12)


def test_crr_converges_to_black_scholes():
    for option_type in ("call", "put"):
        exact = bs(S0, K, T, R, SIGMA, option_type, Q)
        assert crr_price(S0, K, T, R, SIGMA, option_type, q=Q, n_steps=2000) == pytest.approx(exact, abs=5e-3)


def test_american_call_without_dividends_equals_european():
    eu = crr_price(S0, K, T, R, SIGMA, "call", n_steps=500)
    am = crr_price(S0, K, T, R, SIGMA, "call", american=True, n_steps=500)
    assert am == pytest.approx(eu, abs=1e-10)
    assert crr_price(S0, K, T, R, SIGMA, "put", american=True) > crr_price(S0, K, T, R, SIGMA, "put")


def test_longstaff_schwartz_matches_tree():
    # Longstaff & Schwartz (2001), Table 1: S=36, K=40, r=0.06, sigma=0.2, T=1 -> 4.478
    gbm = GBM(36.0, 0.06, 0.2)
    tree = crr_price(36.0, 40.0, 1.0, 0.06, 0.2, "put", american=True, n_steps=2000)
    lsm = longstaff_schwartz(
        gbm,
        lambda s: np.maximum(40.0 - s, 0.0),
        1.0,
        r=0.06,
        n_steps=50,
        n_paths=100_000,
        seed=12,
        antithetic=True,
    )
    assert abs(lsm.price - 4.478) < 0.03
    assert abs(lsm.price - tree) < 0.03
    poly = longstaff_schwartz(
        gbm,
        lambda s: np.maximum(40.0 - s, 0.0),
        1.0,
        r=0.06,
        n_steps=50,
        n_paths=50_000,
        seed=13,
        basis="poly",
        degree=2,
    )
    assert abs(poly.price - tree) < 0.05


def test_validation():
    with pytest.raises(ValueError):
        Digital(K, "straddle")
    with pytest.raises(ValueError):
        crr_price(S0, K, T, R, SIGMA, "forward")
    with pytest.raises(ValueError):
        price_mc(GBM(S0, R, SIGMA), EuropeanCall(K), T, n_paths=1000, seed=0).conf_int(1.5)
