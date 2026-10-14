import numpy as np
import pytest

from stochPathSim.analytics.rates import (
    cir_zcb,
    g2pp_zcb,
    hull_white_caplet,
    hull_white_zcb,
    hull_white_zcb_option,
    vasicek_zcb,
    vasicek_zcb_option,
)
from stochPathSim.curves import ZeroCurve

CURVE = ZeroCurve([0.5, 1.0, 2.0, 5.0, 10.0], [0.02, 0.025, 0.03, 0.035, 0.04])


def test_hull_white_bond_options_match_quantlib():
    # QuantLib 1.44: HullWhite(flat 3% continuous, a=0.1, sigma=0.01).discountBondOption(type, K, 1, 5)
    for strike, call, put in (
        (0.85, 0.03689981178987056, 0.0010705388810446775),
        (0.90, 0.0056695544344894655, 0.018362558203088963),
    ):
        args = (1.0, 5.0, strike, 0.1, 0.01, 0.03)
        assert hull_white_zcb_option(*args, "call") == pytest.approx(call, rel=1e-10)
        assert hull_white_zcb_option(*args, "put") == pytest.approx(put, rel=1e-10)


def test_vasicek_matches_quantlib():
    # QuantLib 1.44: Vasicek(r0=0.03, a=0.5, b=0.05, sigma=0.01)
    assert vasicek_zcb(0.03, 0.0, 2.0, 0.5, 0.05, 0.01) == pytest.approx(0.9280701641344499, rel=1e-12)
    assert vasicek_zcb_option(0.03, 1.0, 3.0, 0.9, 0.5, 0.05, 0.01) == pytest.approx(0.018387857918337303,
                                                                                     rel=1e-10)


@pytest.mark.parametrize(
    "price, p_t, p_s",
    [
        (lambda K, kind: hull_white_zcb_option(2.0, 7.0, K, 0.1, 0.01, CURVE, kind),
         CURVE.discount(2.0), CURVE.discount(7.0)),
        (lambda K, kind: vasicek_zcb_option(0.03, 2.0, 7.0, K, 0.5, 0.05, 0.01, kind),
         vasicek_zcb(0.03, 0.0, 2.0, 0.5, 0.05, 0.01), vasicek_zcb(0.03, 0.0, 7.0, 0.5, 0.05, 0.01)),
    ],
    ids=["hull-white", "vasicek"],
)
def test_bond_option_put_call_parity(price, p_t, p_s):
    for K in (0.7, 0.8, 0.9):
        assert price(K, "call") - price(K, "put") == pytest.approx(p_s - K * p_t, abs=1e-12)


def test_hull_white_bond_price_at_time_zero_is_the_curve():
    r0 = CURVE.forward_rate(0.0)
    T = np.array([0.5, 1.0, 3.0, 8.0])
    assert np.allclose(hull_white_zcb(r0, 0.0, T, 0.1, 0.01, CURVE), CURVE.discount(T), rtol=1e-8)


def test_g2pp_bond_price_at_time_zero_is_the_curve():
    T = np.array([0.5, 2.0, 10.0])
    assert np.allclose(g2pp_zcb(0.0, 0.0, 0.0, T, 0.5, 0.1, 0.01, 0.008, -0.6, CURVE), CURVE.discount(T))


def test_vasicek_and_cir_bond_limits():
    assert vasicek_zcb(0.03, 1.0, 1.0, 0.5, 0.05, 0.01) == pytest.approx(1.0)
    assert cir_zcb(0.03, 1.0, 1.0, 0.5, 0.05, 0.1) == pytest.approx(1.0)
    # with almost no volatility the bond is the deterministic discount factor
    kappa, theta, r0, T = 0.5, 0.05, 0.03, 4.0
    integral = theta * T + (r0 - theta) * (1 - np.exp(-kappa * T)) / kappa
    assert vasicek_zcb(r0, 0.0, T, kappa, theta, 1e-8) == pytest.approx(np.exp(-integral), rel=1e-9)
    assert cir_zcb(r0, 0.0, T, kappa, theta, 1e-4) == pytest.approx(np.exp(-integral), rel=1e-7)


def test_caplet_floorlet_parity():
    T1, T2, K, a, sigma = 1.0, 1.5, 0.03, 0.1, 0.01
    cap = hull_white_caplet(T1, T2, K, a, sigma, CURVE, notional=1e6)
    floor = hull_white_caplet(T1, T2, K, a, sigma, CURVE, notional=1e6, floorlet=True)
    tau = T2 - T1
    fra = 1e6 * (CURVE.discount(T1) - CURVE.discount(T2) * (1 + K * tau))
    assert cap - floor == pytest.approx(fra, abs=1e-6)
    assert cap > 0 and floor > 0


def test_options_are_vectorised():
    K = np.array([0.8, 0.85, 0.9])
    prices = hull_white_zcb_option(1.0, 5.0, K, 0.1, 0.01, CURVE)
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)


def test_invalid_option_type():
    with pytest.raises(ValueError):
        hull_white_zcb_option(1.0, 5.0, 0.9, 0.1, 0.01, CURVE, "straddle")
