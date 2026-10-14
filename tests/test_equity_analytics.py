import numpy as np
import pytest

from stochPathSim.analytics import (
    bachelier_implied_vol,
    bachelier_price,
    black76_price,
    bs_greeks,
    bs_implied_vol,
    bs_price,
    heston_price,
    merton_price,
    sabr_implied_vol,
)

# Reference values from QuantLib 1.44: AnalyticHestonEngine and sabrVolatility / shiftedSabrVolatility.
HESTON_REFERENCES = [
    # S, K, r, q, v0, kappa, theta, xi, rho, call, put (T = 1)
    (100, 100, 0.05, 0.00, 0.04, 2.0, 0.04, 0.3, -0.7, 10.394218565150172, 5.517161015221563),
    (100, 110, 0.03, 0.01, 0.05, 1.5, 0.06, 0.5, -0.5, 5.048746677175362, 12.792771992594462),
    (100, 90, 0.03, 0.01, 0.05, 1.5, 0.06, 0.5, -0.5, 15.799096983427395, 4.13421162787633),
]

SABR_REFERENCES = [
    # F, K, T, alpha, beta, rho, nu, vol
    (0.03, 0.03, 2.0, 0.03, 0.5, -0.3, 0.4, 0.176408597795153),
    (0.03, 0.02, 2.0, 0.03, 0.5, -0.3, 0.4, 0.23302174818090077),
    (0.03, 0.045, 2.0, 0.03, 0.5, -0.3, 0.4, 0.16172518854403775),
    (100.0, 110.0, 1.0, 0.25, 1.0, -0.5, 0.6, 0.2372917536519652),
]


def test_bs_put_call_parity():
    S, K, T, r, q, sigma = 100.0, np.array([80.0, 100.0, 120.0]), 0.75, 0.04, 0.015, 0.25
    call = bs_price(S, K, T, r, sigma, "call", q)
    put = bs_price(S, K, T, r, sigma, "put", q)
    np.testing.assert_allclose(call - put, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-12)


def test_bs_known_value():
    # Hull, Options, Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5
    assert bs_price(42, 40, 0.5, 0.10, 0.2, "call") == pytest.approx(4.7594, abs=1e-4)
    assert bs_price(42, 40, 0.5, 0.10, 0.2, "put") == pytest.approx(0.8086, abs=1e-4)


def test_bs_degenerate_cases_return_intrinsic():
    assert bs_price(110.0, 100.0, 0.0, 0.05, 0.2) == pytest.approx(10.0)
    assert bs_price(100.0, 100.0, 1.0, 0.05, 0.0, "call") == pytest.approx(100 - 100 * np.exp(-0.05))
    assert bs_price(100.0, 120.0, 1.0, 0.05, 0.0, "call") == 0.0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bs_greeks_match_finite_differences(option_type):
    S, K, T, r, sigma, q = 100.0, 105.0, 0.8, 0.03, 0.22, 0.01
    g = bs_greeks(S, K, T, r, sigma, option_type, q)

    def price(**kw):
        p = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        p.update(kw)
        return bs_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], option_type, p["q"])

    h = 1e-4
    assert g["delta"] == pytest.approx((price(S=S + h) - price(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((price(S=S + 1e-2) - 2 * price() + price(S=S - 1e-2)) / 1e-4, rel=1e-4)
    assert g["vega"] == pytest.approx((price(sigma=sigma + h) - price(sigma=sigma - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((price(r=r + h) - price(r=r - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(price(T=T + h) - price(T=T - h)) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bs_implied_vol_round_trip(option_type):
    strikes = np.array([70.0, 90.0, 100.0, 115.0, 140.0])
    vols = np.array([0.45, 0.3, 0.25, 0.22, 0.35])
    prices = bs_price(100.0, strikes, 1.3, 0.02, vols, option_type, 0.01)
    implied = bs_implied_vol(prices, 100.0, strikes, 1.3, 0.02, option_type, 0.01)
    np.testing.assert_allclose(implied, vols, atol=1e-8)


def test_bs_implied_vol_outside_bounds_is_nan():
    assert np.isnan(bs_implied_vol(150.0, 100.0, 100.0, 1.0, 0.0, "call"))
    assert np.isnan(bs_implied_vol(0.5, 100.0, 50.0, 1.0, 0.0, "call"))


def test_black76_consistent_with_bs():
    S, K, T, r, sigma = 100.0, 95.0, 2.0, 0.03, 0.3
    F = S * np.exp(r * T)
    assert black76_price(F, K, T, np.exp(-r * T), sigma) == pytest.approx(bs_price(S, K, T, r, sigma))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bachelier_round_trip_and_parity(option_type):
    F, K, T, df, sn = 0.03, np.array([0.01, 0.025, 0.03, 0.05]), 2.0, 0.94, 0.008
    price = bachelier_price(F, K, T, df, sn, option_type)
    np.testing.assert_allclose(bachelier_implied_vol(price, F, K, T, df, option_type), sn, atol=1e-10)
    parity = bachelier_price(F, K, T, df, sn, "call") - bachelier_price(F, K, T, df, sn, "put")
    np.testing.assert_allclose(parity, df * (F - K), atol=1e-14)


@pytest.mark.parametrize("ref", HESTON_REFERENCES)
def test_heston_matches_quantlib(ref):
    S, K, r, q, v0, kappa, theta, xi, rho, call, put = ref
    assert heston_price(S, K, 1.0, r, q, v0, kappa, theta, xi, rho, "call") == pytest.approx(call, abs=1e-8)
    assert heston_price(S, K, 1.0, r, q, v0, kappa, theta, xi, rho, "put") == pytest.approx(put, abs=1e-8)


def test_heston_tends_to_black_scholes_with_small_vol_of_vol():
    # constant variance v0 = theta, vanishing vol of vol
    price = heston_price(100, 105, 1.0, 0.03, 0.0, 0.04, 1.0, 0.04, 1e-4, 0.0, "call")
    assert price == pytest.approx(bs_price(100, 105, 1.0, 0.03, 0.2), abs=1e-6)


def test_merton_reduces_to_black_scholes_without_jumps():
    no_jumps = merton_price(100, 100, 1.0, 0.05, 0.2, 0.0, -0.1, 0.15)
    assert no_jumps == pytest.approx(bs_price(100, 100, 1.0, 0.05, 0.2))


def test_merton_put_call_parity_and_jump_premium():
    args = (100, 100, 1.0, 0.05, 0.2, 0.5, -0.1, 0.15)
    call, put = merton_price(*args, "call"), merton_price(*args, "put")
    assert call - put == pytest.approx(100 - 100 * np.exp(-0.05), abs=1e-10)
    # jumps add variance, so an at-the-money option is worth more than under Black-Scholes
    assert call > bs_price(100, 100, 1.0, 0.05, 0.2)


@pytest.mark.parametrize("ref", SABR_REFERENCES)
def test_sabr_matches_quantlib(ref):
    F, K, T, alpha, beta, rho, nu, vol = ref
    assert sabr_implied_vol(F, K, T, alpha, beta, rho, nu) == pytest.approx(vol, abs=1e-12)


def test_shifted_sabr_matches_quantlib():
    vol = sabr_implied_vol(0.001, -0.002, 1.5, 0.01, 0.5, -0.2, 0.3, shift=0.01)
    assert vol == pytest.approx(0.12354722101299205, abs=1e-12)


def test_sabr_vectorised_and_continuous_at_the_money():
    strikes = np.array([0.03 - 1e-9, 0.03, 0.03 + 1e-9])
    vols = sabr_implied_vol(0.03, strikes, 2.0, 0.03, 0.5, -0.3, 0.4)
    assert np.ptp(vols) < 1e-8
