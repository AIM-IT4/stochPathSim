import numpy as np
import pytest

import stochPathSim as sps
from stochPathSim.analytics import bs_price, heston_price
from stochPathSim.pricing import EuropeanCall
from stochPathSim.validation import (
    ValidationReport,
    compare_models,
    convergence_plot,
    convergence_study,
    discretisation_plot,
    discretisation_study,
)

HESTON = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)


def test_convergence_error_shrinks_like_root_n():
    gbm = sps.GBM(100.0, 0.05, 0.2)
    ref = bs_price(100.0, 100.0, 1.0, 0.05, 0.2)
    df = convergence_study(
        gbm,
        EuropeanCall(100),
        1.0,
        ref,
        n_paths_list=(1_000, 16_000, 256_000),
        n_steps=1,
        seed=1,
        discount=0.05,
    )
    assert list(df.columns) == ["n_paths", "price", "stderr", "error", "abs_error", "z_score"]
    # standard error falls by 4 for every 16x more paths
    np.testing.assert_allclose(df.stderr.iloc[0] / df.stderr.iloc[1], 4.0, rtol=0.1)
    np.testing.assert_allclose(df.stderr.iloc[1] / df.stderr.iloc[2], 4.0, rtol=0.1)
    assert np.all(np.abs(df.z_score) < 4)
    fig = convergence_plot(df)
    assert fig.layout.xaxis2.type == "log" and fig.layout.yaxis2.type == "log"


def test_discretisation_bias_decreases_for_heston_euler():
    ref = heston_price(100.0, 100.0, 1.0, 0.03, 0.0, **HESTON)
    df = discretisation_study(
        lambda n: sps.Heston(100.0, 0.03, **HESTON, scheme="euler"),
        EuropeanCall(100),
        1.0,
        ref,
        n_steps_list=(2, 4, 8, 16),
        n_paths=60_000,
        seed=3,
        discount=0.03,
    )
    assert np.all(np.diff(df.abs_bias) < 0)
    assert 0.6 < df.attrs["weak_order"] < 1.5  # Euler has weak order 1
    assert "weak order" in discretisation_plot(df).layout.title.text


def test_qe_scheme_has_small_bias_on_coarse_grid():
    ref = heston_price(100.0, 100.0, 1.0, 0.03, 0.0, **HESTON)
    df = discretisation_study(
        sps.Heston(100.0, 0.03, **HESTON),
        EuropeanCall(100),
        1.0,
        ref,
        n_steps_list=(4, 16),
        n_paths=60_000,
        seed=3,
        discount=0.03,
    )
    assert df.abs_bias.iloc[-1] < 3 * df.stderr.iloc[-1] + 0.02


def test_compare_models_table():
    models = {
        "GBM": sps.GBM(100.0, 0.03, 0.2),
        "Merton": sps.MertonJumpDiffusion(100.0, 0.03, 0.15, 1.0, -0.1, 0.15),
    }
    comp = compare_models(models, T=1.0, n_steps=12, n_paths=20_000, seed=5)
    assert list(comp.table.index) == ["GBM", "Merton"]
    assert {"mean", "std", "skew", "excess_kurtosis", "q1", "q99"} <= set(comp.table.columns)
    # both are martingales after discounting at the same rate
    np.testing.assert_allclose(comp.table["mean"], 100 * np.exp(0.03), rtol=0.01)
    # negative jumps fatten the left tail
    assert comp.table.loc["Merton", "q1"] < comp.table.loc["GBM", "q1"]
    assert len(comp.figure.data) == 2


def _backtest_data(model_vol, seed=0, n_dates=400):
    rng = np.random.default_rng(seed)
    h = 10 / 252
    realised = 100 * np.exp(rng.normal(-0.5 * 0.25**2 * h, 0.25 * np.sqrt(h), n_dates))
    sims = sps.GBM(100, 0.0, model_vol).simulate(T=h, n_steps=1, n_paths=4000, seed=1).terminal()
    return np.tile(sims, (n_dates, 1)), realised


def test_report_backtest_pass_and_fail():
    good = ValidationReport("good")
    out = good.add_backtest(*_backtest_data(0.25), level=0.99)
    assert good.passed and out["ad_pvalue"] > 0.05
    bad = ValidationReport("bad")
    bad.add_backtest(*_backtest_data(0.12), level=0.99)
    assert not bad.passed
    assert (bad.summary().result == "FAIL").sum() >= 2


def test_report_writes_self_contained_html(tmp_path):
    ref = bs_price(100.0, 100.0, 1.0, 0.05, 0.2)
    rep = ValidationReport("GBM validation")
    rep.add_convergence(
        sps.GBM(100.0, 0.05, 0.2),
        EuropeanCall(100),
        1.0,
        ref,
        n_paths_list=(1_000, 4_000),
        n_steps=1,
        seed=1,
        discount=0.05,
    )
    rep.add_comparison({"GBM": sps.GBM(100.0, 0.05, 0.2)}, T=1.0, n_steps=4, n_paths=2_000, seed=2)
    rep.add_backtest(*_backtest_data(0.25, n_dates=100), level=0.95)
    summary = rep.summary()
    assert list(summary.columns) == ["section", "check", "value", "threshold", "result"]
    path = rep.to_html(tmp_path / "report.html")
    text = path.read_text(encoding="utf-8")
    for title in ("GBM validation", "Monte Carlo convergence", "Model comparison", "Risk-factor backtest"):
        assert title in text
    assert text.count('class="plotly-graph-div"') == 3
    assert "plotly.js" in text.lower() or "Plotly.newPlot" in text
    assert path.stat().st_size > 1_000_000  # plotly.js is inlined, so the file works offline
    cdn = ValidationReport("cdn", include_plotlyjs="cdn")
    cdn.add_figure(sps.GBM(1, 0, 0.1).simulate(1, 2, 4, seed=0).plot(), "paths")
    assert cdn.to_html(tmp_path / "cdn.html").stat().st_size < 200_000
    with pytest.raises(ValueError):
        ValidationReport(include_plotlyjs="nope")
