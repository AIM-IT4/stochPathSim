import numpy as np
import pytest
from scipy import stats

from stochPathSim.processes import GBM
from stochPathSim.risk import (
    ExposureProfile,
    anderson_darling_uniform,
    christoffersen_independence,
    collateralised_exposure,
    cramer_von_mises_uniform,
    cva_from_exposure,
    effective_ee,
    effective_epe,
    exceedance_counts,
    expected_exposure,
    expected_negative_exposure,
    expected_positive_exposure,
    kupiec_pof,
    monte_carlo_var,
    path_statistics,
    pit_values,
    potential_future_exposure,
    var_es,
)


@pytest.fixture
def brownian_mtm():
    rng = np.random.default_rng(7)
    times = np.linspace(0.0, 2.0, 9)
    s = 10.0
    z = rng.standard_normal((200_000, times.size))
    return times, s, s * np.sqrt(times) * z


# --- path statistics -------------------------------------------------------------


def test_path_statistics_columns_and_values():
    res = GBM(100.0, 0.05, 0.2).simulate(T=1.0, n_steps=4, n_paths=20_000, seed=1)
    df = path_statistics(res)
    assert list(df.columns) == ["mean", "std", "p_level", "q01", "q10", "q90", "q99"]
    assert df.index.name == "time"
    np.testing.assert_allclose(df.index.values, res.times)
    # at t=0 all paths equal the initial value: none strictly above it
    assert df["p_level"].iloc[0] == 0.0
    assert df["mean"].iloc[-1] == pytest.approx(105.127, rel=0.01)
    assert (df["q01"] <= df["q10"]).all() and (df["q90"] <= df["q99"]).all()
    # P(S_1 > 100) = N(d2) with d2 = (mu - sigma^2/2)/sigma
    assert df["p_level"].iloc[-1] == pytest.approx(stats.norm.cdf(0.15), abs=0.01)


def test_path_statistics_array_and_reference():
    paths = np.array([[1.0, 2.0, 3.0], [1.0, 0.0, 5.0]])
    df = path_statistics(paths, reference=1.5, quantiles=(0.5,))
    np.testing.assert_allclose(df["p_level"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(df["q50"], [1.0, 1.0, 4.0])


def test_exceedance_counts_matches_expected_rate():
    rng = np.random.default_rng(3)
    sims = rng.standard_normal((50_000, 50))
    realised = rng.standard_normal((200, 50))  # correct model
    out = exceedance_counts(sims, realised, lower=0.05, upper=0.95)
    assert out["n_obs"] == 10_000
    assert out["upper"] == pytest.approx(out["expected_upper"], rel=0.1)
    assert out["lower"] == pytest.approx(out["expected_lower"], rel=0.1)
    wide = exceedance_counts(sims, 3.0 * realised, lower=0.05, upper=0.95)
    assert wide["upper"] > 3 * out["upper"]


def test_exceedance_counts_validates():
    with pytest.raises(ValueError):
        exceedance_counts(np.zeros((10, 3)), np.zeros(4))
    with pytest.raises(ValueError):
        exceedance_counts(np.zeros((10, 3)), np.zeros(3), lower=0.9, upper=0.1)


# --- exposure --------------------------------------------------------------------


def test_ee_and_pfe_for_brownian_mtm(brownian_mtm):
    times, s, mtm = brownian_mtm
    scale = s * np.sqrt(times)
    ee = expected_exposure(mtm)
    np.testing.assert_allclose(ee, scale / np.sqrt(2 * np.pi), rtol=0.01, atol=1e-12)
    ene = expected_negative_exposure(mtm)
    np.testing.assert_allclose(ene, -scale / np.sqrt(2 * np.pi), rtol=0.01, atol=1e-12)
    pfe = potential_future_exposure(mtm, 0.95)
    np.testing.assert_allclose(pfe, scale * stats.norm.ppf(0.95), rtol=0.01, atol=1e-12)


def test_epe_and_effective_measures():
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    ee = np.array([0.0, 2.0, 3.0, 1.0, 4.0, 10.0])
    eff = effective_ee(ee)
    np.testing.assert_allclose(eff, [0, 2, 3, 3, 4, 10])
    assert np.all(np.diff(eff) >= 0)
    # left-point time average over the whole grid
    assert expected_positive_exposure(ee, times) == pytest.approx(
        (0 * 0.25 + 2 * 0.25 + 3 * 0.25 + 1 * 0.25 + 4 * 0.5) / 1.5
    )
    # effective EPE uses the first year only
    assert effective_epe(ee, times, horizon=1.0) == pytest.approx((0 + 2 + 3 + 3) * 0.25)


def test_collateral_reduces_exposure():
    # Brownian mark-to-market paths on a fine grid
    rng = np.random.default_rng(8)
    dt, n = 0.01, 200
    increments = 10.0 * np.sqrt(dt) * rng.standard_normal((50_000, n))
    mtm = np.concatenate([np.zeros((50_000, 1)), np.cumsum(increments, axis=1)], axis=1)
    uncoll = expected_exposure(mtm)
    mpor = 2
    coll = expected_exposure(collateralised_exposure(mtm, threshold=0.0, margin_period_steps=mpor))
    # before the first margin call nothing changes; afterwards the exposure is that of
    # the value change over the margin period: 10 sqrt(mpor dt) / sqrt(2 pi)
    np.testing.assert_allclose(coll[: mpor + 1], uncoll[: mpor + 1])
    assert np.all(coll[mpor + 1 :] < uncoll[mpor + 1 :])
    np.testing.assert_allclose(coll[mpor + 1 :], 10.0 * np.sqrt(mpor * dt) / np.sqrt(2 * np.pi), rtol=0.03)
    # perfect, instantaneous collateral with zero threshold removes all exposure
    np.testing.assert_allclose(expected_exposure(collateralised_exposure(mtm)), 0.0, atol=1e-12)
    # an infinite threshold means no collateral
    np.testing.assert_allclose(collateralised_exposure(mtm, threshold=1e12, margin_period_steps=1), mtm)


def test_exposure_profile_bundle(brownian_mtm):
    times, _, mtm = brownian_mtm
    prof = ExposureProfile.from_mtm(mtm, times, pfe_level=0.99)
    df = prof.to_frame()
    assert list(df.columns) == ["EE", "ENE", "PFE99", "EffectiveEE"]
    assert prof.max_pfe == pytest.approx(df["PFE99"].max())
    assert prof.epe > 0


def test_cva_flat_exposure_closed_form():
    times = np.linspace(0.0, 5.0, 5001)
    ee = np.full(times.size, 100.0)
    h, r, rec = 0.02, 0.03, 0.4
    cva = cva_from_exposure(ee, times, hazard_rate=h, recovery=rec, discount_rate=r)
    exact = (1 - rec) * 100.0 * h / (h + r) * (1 - np.exp(-(h + r) * 5.0))
    assert cva == pytest.approx(exact, rel=1e-3)
    surv = np.exp(-h * times)
    assert cva_from_exposure(ee, times, survival=surv, recovery=rec, discount_rate=r) == pytest.approx(cva)


# --- VaR -------------------------------------------------------------------------


def test_parametric_var_equals_normal_quantile():
    rng = np.random.default_rng(11)
    pnl = rng.normal(1.0, 2.0, 100_000)
    var, es = var_es(pnl, 0.99, "parametric")
    mu, sd = -pnl.mean(), pnl.std(ddof=1)
    assert var == pytest.approx(mu + sd * stats.norm.ppf(0.99))
    assert es == pytest.approx(mu + sd * stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01)
    hvar, hes = var_es(pnl, 0.99, "historical")
    assert hvar == pytest.approx(var, rel=0.02) and hes > hvar
    cvar, _ = var_es(pnl, 0.99, "cornish_fisher")
    assert cvar == pytest.approx(var, rel=0.02)
    with pytest.raises(ValueError):
        var_es(pnl, 0.99, "bogus")


def test_monte_carlo_var_gbm():
    res = GBM(100.0, 0.0, 0.2).simulate(T=0.04, n_steps=1, n_paths=200_000, seed=5)
    var, es = monte_carlo_var(res.terminal(), 100.0, 0.99)
    expected = 100.0 * (1 - np.exp(-0.5 * 0.04 * 0.04 + 0.2 * 0.2 * stats.norm.ppf(0.01)))
    assert var == pytest.approx(expected, rel=0.02)
    assert es > var


# --- backtests -------------------------------------------------------------------


def test_kupiec_pof():
    n, x, p = 250, 8, 0.01
    lr, pval = kupiec_pof(n, x, 0.99)
    ph = x / n
    expected = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
    assert lr == pytest.approx(expected)
    assert lr == pytest.approx(7.7336, abs=1e-3)  # rejects at 1%: chi2(1) 99% quantile is 6.63
    assert pval < 0.01
    assert pval == pytest.approx(stats.chi2.sf(expected, 1))
    assert kupiec_pof(1000, 10, 0.99)[0] == pytest.approx(0.0, abs=1e-12)
    assert kupiec_pof(250, 0, 0.99)[0] > 0


def test_christoffersen_independence():
    rng = np.random.default_rng(2)
    indep = rng.random(5000) < 0.05
    assert christoffersen_independence(indep)[1] > 0.01
    clustered = np.zeros(5000, dtype=bool)
    for start in range(0, 5000, 100):
        clustered[start : start + 5] = True
    assert christoffersen_independence(clustered)[1] < 1e-6


def test_pit_uniform_for_correct_model_and_rejects_wrong_vol():
    rng = np.random.default_rng(4)
    n_dates, n_sims = 500, 2000
    true_sd = 0.2
    realised = rng.normal(0.0, true_sd, n_dates)
    good = rng.normal(0.0, true_sd, (n_dates, n_sims))
    bad = rng.normal(0.0, 0.5 * true_sd, (n_dates, n_sims))
    pit_good = pit_values(good, realised)
    pit_bad = pit_values(bad, realised)
    assert pit_good.shape == (n_dates,)
    assert anderson_darling_uniform(pit_good)[1] > 0.01
    assert cramer_von_mises_uniform(pit_good)[1] > 0.01
    assert anderson_darling_uniform(pit_bad)[1] < 1e-4
    assert cramer_von_mises_uniform(pit_bad)[1] < 1e-4


def test_pit_shape_check():
    with pytest.raises(ValueError):
        pit_values(np.zeros((3, 10)), np.zeros(4))
