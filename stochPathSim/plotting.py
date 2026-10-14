"""Matplotlib plots of simulated paths and exposure profiles.

Matplotlib is optional: ``pip install stochPathSim[plot]``. Functions return
the Axes and never call ``plt.show()``.
"""

from __future__ import annotations

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("plotting needs matplotlib: pip install stochPathSim[plot]") from exc
    return plt


def _axes(ax):
    if ax is not None:
        return ax
    plt = _pyplot()
    _, ax = plt.subplots(figsize=(10, 5))
    return ax


def plot_paths(result, name=None, n_paths: int = 20, ax=None, quantiles=(0.01, 0.99), title=None):
    """Plot a sample of paths, the mean and a quantile band."""
    ax = _axes(ax)
    paths = result.get(name)
    t = result.times
    k = min(int(n_paths), paths.shape[0])
    if k:
        ax.plot(t, paths[:k].T, lw=0.8, alpha=0.6)
    ax.plot(t, paths.mean(axis=0), color="black", lw=2, label="mean")
    if quantiles:
        lo, hi = np.quantile(paths, quantiles, axis=0)
        ax.fill_between(
            t, lo, hi, color="grey", alpha=0.2, label=f"{quantiles[0]:.0%}-{quantiles[1]:.0%} band"
        )
    ax.set_xlabel("time (years)")
    ax.set_ylabel(name or result.primary)
    ax.set_title(title or f"{paths.shape[0]:,} simulated paths")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="best")
    return ax


def plot_fan(result, name=None, quantiles=(0.01, 0.05, 0.25, 0.75, 0.95, 0.99), ax=None, title=None):
    """Fan chart: nested symmetric quantile bands around the median."""
    ax = _axes(ax)
    paths = result.get(name)
    t = result.times
    qs = sorted(quantiles)
    values = np.quantile(paths, qs, axis=0)
    n = len(qs) // 2
    for i in range(n):
        ax.fill_between(
            t,
            values[i],
            values[-1 - i],
            color="tab:blue",
            alpha=0.15 + 0.6 * i / max(n, 1),
            lw=0,
            label=f"{qs[i]:.0%}-{qs[-1 - i]:.0%}",
        )
    ax.plot(t, np.median(paths, axis=0), color="tab:blue", lw=2, label="median")
    ax.set_xlabel("time (years)")
    ax.set_ylabel(name or result.primary)
    ax.set_title(title or "Simulated distribution")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="best")
    return ax


def plot_exposure(profile, ax=None, title=None):
    """Plot EE, effective EE, PFE and ENE of an :class:`ExposureProfile`."""
    ax = _axes(ax)
    t = profile.times
    ax.plot(t, profile.ee, lw=2, label="EE")
    ax.plot(t, profile.effective_ee, lw=1.5, ls="--", label="Effective EE")
    ax.plot(t, profile.pfe, lw=2, label=f"PFE {profile.pfe_level:.0%}")
    ax.plot(t, profile.ene, lw=1.5, label="ENE")
    ax.axhline(profile.epe, color="grey", ls=":", label=f"EPE = {profile.epe:.4g}")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("exposure")
    ax.set_title(title or "Exposure profile")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="best")
    return ax
