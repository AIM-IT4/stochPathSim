import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from stochPathSim import plotting  # noqa: E402
from stochPathSim.processes import GBM  # noqa: E402
from stochPathSim.risk import ExposureProfile  # noqa: E402


@pytest.fixture
def result():
    return GBM(100.0, 0.05, 0.2).simulate(T=1.0, n_steps=20, n_paths=500, seed=1)


def test_plot_paths(result):
    ax = plotting.plot_paths(result, n_paths=5)
    assert len(ax.lines) == 6  # 5 paths + mean
    assert ax.get_ylabel() == "spot"
    plt.close("all")


def test_plot_paths_uses_given_axes(result):
    _, ax = plt.subplots()
    assert plotting.plot_paths(result, ax=ax, quantiles=None) is ax
    plt.close("all")


def test_result_plot_method(result):
    ax = result.plot(n_paths=3)
    assert len(ax.lines) == 4
    plt.close("all")


def test_plot_fan(result):
    ax = plotting.plot_fan(result)
    assert len(ax.collections) == 3
    plt.close("all")


def test_plot_exposure():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 1, 11)
    prof = ExposureProfile.from_mtm(np.sqrt(times) * rng.standard_normal((1000, 11)), times)
    ax = plotting.plot_exposure(prof)
    assert len(ax.lines) == 5
    plt.close("all")
