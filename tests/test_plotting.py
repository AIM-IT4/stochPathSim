import numpy as np
import plotly.graph_objects as go
import pytest

from stochPathSim import plotting
from stochPathSim.processes import GBM
from stochPathSim.risk import ExposureProfile


@pytest.fixture
def result():
    return GBM(100.0, 0.05, 0.2).simulate(T=1.0, n_steps=20, n_paths=500, seed=1)


def test_plot_paths(result):
    fig = plotting.plot_paths(result, n_paths=5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2 + 5 + 1  # band (2) + 5 paths + mean
    assert fig.layout.yaxis.title.text == "spot"


def test_plot_paths_accepts_ax_for_compatibility(result):
    fig = plotting.plot_paths(result, ax=object(), quantiles=None, n_paths=3)
    assert len(fig.data) == 4


def test_result_plot_method(result):
    assert len(result.plot(n_paths=3).data) == 2 + 3 + 1


def test_plot_fan(result):
    fig = plotting.plot_fan(result)
    assert len(fig.data) == 3 * 2 + 1  # three bands + median


def test_plot_exposure():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 1, 11)
    prof = ExposureProfile.from_mtm(np.sqrt(times) * rng.standard_normal((1000, 11)), times)
    fig = plotting.plot_exposure(prof)
    assert {t.name for t in fig.data} >= {"EE", "Effective EE", "ENE", "PFE 95%"}
