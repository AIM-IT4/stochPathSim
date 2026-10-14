import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest
from scipy import stats

import stochPathSim as sps
from stochPathSim import viz
from stochPathSim.risk import ExposureProfile


@pytest.fixture(scope="module")
def heston():
    model = sps.Heston(100.0, 0.03, 0.04, 1.5, 0.04, 0.6, -0.7)
    return model.simulate(T=1.0, n_steps=20, n_paths=1000, seed=1)


def test_theme_is_registered_and_switchable():
    assert "stochpathsim" in pio.templates and "stochpathsim_dark" in pio.templates
    assert viz.set_theme("dark") == "stochpathsim_dark"
    assert (
        viz.paths(sps.GBM(1, 0, 0.1).simulate(1, 2, 4, seed=0)).layout.template
        == pio.templates["stochpathsim_dark"]
    )
    assert viz.set_theme("light") == "stochpathsim"
    with pytest.raises(ValueError):
        viz.set_theme("neon")


def test_paths_and_fan(heston):
    fig = viz.paths(heston, n_paths=10, quantiles=(0.05, 0.95))
    assert isinstance(fig, go.Figure) and len(fig.data) == 2 + 10 + 1
    assert len(viz.paths(heston, "variance", n_paths=0, quantiles=None, show_mean=False).data) == 0
    assert len(viz.fan(heston, "variance").data) == 7


def test_path_explorer_controls(heston):
    fig = viz.path_explorer(heston, max_paths=10)
    assert len(fig.data) == 2 * (10 + 1)
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == ["spot", "variance"]
    assert [s.label for s in fig.layout.sliders[0].steps] == ["1", "5", "10"]
    # the slider only changes opacity, the dropdown visibility
    assert len(fig.layout.sliders[0].steps[0].args[0]["opacity"]) == len(fig.data)


def test_terminal_distribution_with_pdf():
    gbm = sps.GBM(100.0, 0.0, 0.2)
    res = gbm.simulate(1.0, 4, 5000, seed=2)
    pdf = stats.lognorm(s=0.2, scale=100 * np.exp(-0.02)).pdf
    fig = viz.terminal_distribution(res, pdf=pdf)
    assert [type(t).__name__ for t in fig.data] == ["Histogram", "Scatter"]
    assert res.distribution(t=0.5).data[0].type == "histogram"


@pytest.mark.parametrize(
    "kind, trace", [("heatmap", "heatmap"), ("surface", "surface"), ("animation", "bar")]
)
def test_density_evolution(heston, kind, trace):
    fig = viz.density_evolution(heston, kind=kind, n_times=10)
    assert fig.data[0].type == trace
    if kind == "animation":
        assert len(fig.frames) == 10 and fig.layout.sliders
    with pytest.raises(ValueError):
        viz.density_evolution(heston, kind="pie")


def test_correlation_heatmap():
    corr = [[1.0, 0.6], [0.6, 1.0]]
    res = sps.MultiAssetGBM([100, 100], [0.0, 0.0], [0.2, 0.3], corr).simulate(1.0, 50, 2000, seed=3)
    z = np.array(viz.correlation_heatmap(res).data[0].z)
    assert z.shape == (2, 2) and abs(z[0, 1] - 0.6) < 0.05
    with pytest.raises(ValueError):
        viz.correlation_heatmap(sps.GBM(1, 0, 0.1).simulate(1, 2, 4, seed=0))


def test_risk_dashboards():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 1, 11)
    prof = ExposureProfile.from_mtm(np.sqrt(times) * rng.standard_normal((2000, 11)), times)
    assert len(viz.exposure_dashboard(prof).data) == 4
    pit = rng.uniform(size=300)
    fig = viz.backtest_dashboard(pit, level=0.95)
    assert len(fig.layout.annotations) >= 3  # three panel titles
    realised = rng.standard_normal(100)
    fig2 = viz.backtest_dashboard(pit[:100], realised=realised, bands=(np.full(100, -2.0), np.full(100, 2.0)))
    assert any(t.name == "realised" for t in fig2.data)
    var_fig = viz.var_plot(rng.standard_normal(10_000), level=0.99)
    assert len(var_fig.layout.shapes) == 2  # VaR and ES lines


def test_rates_charts():
    curve = sps.ZeroCurve([1, 5], [0.03, 0.04])
    res = sps.HullWhite(0.05, 0.01, curve).simulate(5.0, 20, 4000, seed=4)
    fig = viz.curve_fit_check(res, curve)
    mc, ref = np.array(fig.data[2].y), np.array(fig.data[3].y)
    assert np.max(np.abs(mc - ref)) < 2e-3
    assert len(viz.short_rate_fan(res).data) == 7


def test_figures_never_show(monkeypatch, heston):
    monkeypatch.setattr(go.Figure, "show", lambda *a, **k: pytest.fail("show() was called"))
    viz.paths(heston)
    viz.fan(heston)
    viz.density_evolution(heston)
