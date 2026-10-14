"""Plotting helpers kept for backward compatibility; they delegate to :mod:`stochPathSim.viz`.

All functions return an interactive ``plotly.graph_objects.Figure`` and never call ``show()``.
The ``ax`` argument of earlier matplotlib-based versions is accepted and ignored.
"""

from __future__ import annotations

from . import viz


def plot_paths(result, name=None, n_paths: int = 20, ax=None, quantiles=(0.01, 0.99), title=None):
    """Sample paths, the mean and a quantile band."""
    return viz.paths(result, name=name, n_paths=n_paths, quantiles=quantiles, title=title)


def plot_fan(result, name=None, quantiles=(0.01, 0.05, 0.25, 0.75, 0.95, 0.99), ax=None, title=None):
    """Fan chart: nested symmetric quantile bands around the median."""
    return viz.fan(result, name=name, quantiles=quantiles, title=title)


def plot_exposure(profile, ax=None, title=None):
    """EE, Effective EE, PFE and ENE of an :class:`~stochPathSim.risk.ExposureProfile`."""
    return viz.exposure_dashboard(profile, title=title)
