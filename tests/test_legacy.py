"""The version 1 interface keeps working on top of the new engines."""

import numpy as np
import pytest

from stochPathSim.bdt_model import BlackDermanToyModel
from stochPathSim.black_scholes import BlackScholes
from stochPathSim.cir import CoxIngersollRossModel
from stochPathSim.gbm import GeometricBrownianMotion
from stochPathSim.heston_model import HestonModel
from stochPathSim.ho_lee_model import HoLeeModel
from stochPathSim.hull_white import HullWhiteModel
from stochPathSim.statistics_utils import calculate_statistics, display_statistics, plot_simulation
from stochPathSim.vasicek import VasicekModel


def test_gbm_shapes_and_start():
    t, s = GeometricBrownianMotion(mu=0.1, sigma=0.2, s0=100, T=1, n=253, paths=500, seed=1).simulate()
    assert s.shape == (500, 253) and t.shape == (253,)
    assert t[0] == 0.0 and t[-1] == 1.0
    assert np.all(s[:, 0] == 100.0)


def test_heston_shapes():
    s, v = HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, s0=100,
                       T=1, n=100, paths=400, seed=2).simulate()
    assert s.shape == v.shape == (400, 100)
    assert np.all(v >= 0.0) and np.all(s[:, 0] == 100.0)


@pytest.mark.parametrize("model", [
    VasicekModel(r0=0.03, kappa=1.0, theta=0.04, sigma=0.01, T=2, n=50, paths=300, seed=3),
    CoxIngersollRossModel(r0=0.03, kappa=1.0, theta=0.04, sigma=0.05, T=2, n=50, paths=300, seed=3),
    HullWhiteModel(r0=0.03, a=0.1, sigma=0.01, T=2, n=50, paths=300, seed=3),
    HoLeeModel(r0=0.03, theta=0.001, sigma=0.01, T=2, n=50, paths=300, seed=3),
    BlackDermanToyModel(r0=0.03, sigma=0.2, T=2, n=50, paths=300, seed=3),
])
def test_short_rate_models(model):
    r = model.simulate()
    assert r.shape == (300, 50)
    np.testing.assert_allclose(r[:, 0], 0.03)


def test_cir_stays_non_negative():
    model = CoxIngersollRossModel(r0=0.01, kappa=0.5, theta=0.02, sigma=0.3, T=5, n=100, paths=500, seed=4)
    r = model.simulate()
    assert np.all(r >= 0.0)


def test_bdt_is_lognormal():
    r = BlackDermanToyModel(r0=0.03, sigma=0.5, T=5, n=60, paths=500, seed=5).simulate()
    assert np.all(r > 0.0)


def test_black_scholes():
    call = BlackScholes.option_price(100, 100, 1, 0.05, 0.2, "call")
    assert call == pytest.approx(10.450583572185565, abs=1e-10)
    mc = BlackScholes.monte_carlo_simulation(100, 100, 1, 0.05, 0.2, paths=200_000, n=2, seed=6)
    assert mc == pytest.approx(call, abs=0.15)
    with pytest.raises(ValueError):
        BlackScholes.option_price(100, 100, 1, 0.05, 0.2, "straddle")


def test_statistics_utils(capsys):
    _, s = GeometricBrownianMotion(mu=0.0, sigma=0.2, s0=100, T=1, n=11, paths=2000, seed=7).simulate()
    stats = calculate_statistics(s, 100)
    quantiles = ["Quantile 99%", "Quantile 90%", "Quantile 10%", "Quantile 1%"]
    assert list(stats.columns) == ["P_LEVEL", "Mean", *quantiles]
    assert {"Upper Exceedance", "Lower Exceedance"} <= set(stats.attrs)
    shown = display_statistics(stats, np.linspace(0, 1, 11))
    assert "Time" in shown.columns and "Time" not in stats.columns
    assert "Exceedance Statistics" in capsys.readouterr().out


def test_plot_simulation_is_not_interactive():
    t, s = GeometricBrownianMotion(mu=0.0, sigma=0.2, s0=100, T=1, n=11, paths=30, seed=8).simulate()
    fig = plot_simulation(t, s, num_paths=5)
    assert len(fig.data) == 5
