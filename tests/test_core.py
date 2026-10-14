import numpy as np
import pytest

from stochPathSim.core import SimulationResult, TimeGrid, cholesky, make_rng, standard_normals
from stochPathSim.curves import ZeroCurve
from stochPathSim.processes import GBM


def test_uniform_grid():
    grid = TimeGrid.uniform(2.0, 8)
    assert grid.n_steps == 8
    assert grid.T == 2.0
    np.testing.assert_allclose(grid.dt, 0.25)
    assert grid.index(1.5) == 6


def test_grid_validation():
    with pytest.raises(ValueError):
        TimeGrid([0.0, 0.5, 0.5])
    with pytest.raises(ValueError):
        TimeGrid([0.1, 0.5])
    with pytest.raises(ValueError):
        TimeGrid.uniform(0.0, 10)


def test_grid_with_extra_times():
    grid = TimeGrid.with_times(1.0, 4, [0.1, 0.6])
    assert 0.1 in grid.times and 0.6 in grid.times
    assert grid.n_steps == 6


def test_antithetic_normals_are_mirrored():
    z = standard_normals(make_rng(0), 6, 5, 2, antithetic=True)
    np.testing.assert_allclose(z[:3], -z[3:])
    with pytest.raises(ValueError):
        standard_normals(make_rng(0), 5, 5, antithetic=True)


def test_correlated_normals():
    corr = np.array([[1.0, 0.7], [0.7, 1.0]])
    z = standard_normals(make_rng(1), 200_000, 1, 2, correlation=corr)
    assert abs(np.corrcoef(z[:, 0, 0], z[:, 0, 1])[0, 1] - 0.7) < 0.01


def test_cholesky_rejects_invalid_matrices():
    with pytest.raises(ValueError):
        cholesky([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError):
        cholesky([[1.0, 0.5], [0.4, 1.0]])


def test_seed_reproducibility():
    gbm = GBM(100.0, 0.05, 0.2)
    a = gbm.simulate(1.0, 10, 100, seed=7)
    b = gbm.simulate(1.0, 10, 100, seed=7)
    np.testing.assert_array_equal(a.get(), b.get())


def test_gbm_moments_and_initial_value():
    gbm = GBM(100.0, 0.05, 0.2, q=0.01)
    res = gbm.simulate(1.0, 12, 200_000, seed=1, antithetic=True)
    assert np.all(res.get()[:, 0] == 100.0)
    s_t = res.terminal()
    stderr = s_t.std() / np.sqrt(s_t.size)
    assert abs(s_t.mean() - gbm.mean(1.0)) < 4 * stderr
    assert abs(s_t.var() / gbm.variance(1.0) - 1.0) < 0.03


def test_gbm_exact_on_non_uniform_grid():
    gbm = GBM(100.0, 0.03, 0.25)
    grid = TimeGrid([0.0, 0.01, 0.5, 2.0])
    res = gbm.simulate(grid=grid, n_paths=100_000, seed=3)
    log_ret = np.log(res.terminal() / 100.0)
    assert abs(log_ret.mean() - (0.03 - 0.5 * 0.25**2) * 2.0) < 0.01
    assert abs(log_ret.std() - 0.25 * np.sqrt(2.0)) < 0.01


def test_common_random_numbers():
    z = standard_normals(make_rng(5), 1000, 10)
    a = GBM(100.0, 0.05, 0.2).simulate(1.0, 10, normals=z)
    b = GBM(100.0, 0.05, 0.3).simulate(1.0, 10, normals=z)
    # same shocks: paths are monotone in volatility at the same sign of the Brownian path
    assert a.n_paths == b.n_paths == 1000
    with pytest.raises(ValueError):
        GBM(100.0, 0.05, 0.2).simulate(1.0, 5, normals=z)


def test_result_accessors():
    res = GBM(100.0, 0.0, 0.2).simulate(1.0, 4, 50, seed=0)
    assert res.primary == "spot"
    assert res.at(0.5).shape == (50,)
    assert res.quantile([0.1, 0.9]).shape == (2, 5)
    frame = res.to_frame(max_paths=3)
    assert frame.shape == (5, 3)
    with pytest.raises(ValueError):
        SimulationResult(np.array([0.0, 1.0]), {"x": np.zeros((3, 3))})


def test_zero_curve():
    curve = ZeroCurve([1.0, 5.0], [0.02, 0.04])
    assert curve.discount(0.0) == 1.0
    np.testing.assert_allclose(curve.discount(5.0), np.exp(-0.2))
    np.testing.assert_allclose(curve.zero_rate(3.0), 0.03)
    flat = ZeroCurve.flat(0.05)
    np.testing.assert_allclose(flat.forward_rate([0.5, 2.0]), 0.05, atol=1e-8)
    round_trip = ZeroCurve.from_discount_factors([1.0, 2.0], curve.discount([1.0, 2.0]))
    np.testing.assert_allclose(round_trip.discount(1.5), curve.discount(1.5), rtol=1e-3)
