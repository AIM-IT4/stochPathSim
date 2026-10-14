"""Version 1 interface, kept for backward compatibility.

Prefer :func:`stochPathSim.risk.path_statistics` and :func:`stochPathSim.risk.exceedance_counts`,
and :func:`stochPathSim.plotting.plot_paths` for plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_statistics(simulations, initial_value):
    """Probability level, mean and quantiles of simulated paths at each time.

    :param simulations: array of shape ``(paths, time steps)``
    :param initial_value: reference level for ``P_LEVEL`` (share of paths above it)
    :return: DataFrame with columns ``P_LEVEL``, ``Mean`` and ``Quantile 99/90/10/1%``.

    ``attrs["Upper Exceedance"]`` and ``attrs["Lower Exceedance"]`` count the times at which the
    initial value lies above the 99% or below the 1% quantile. To backtest a model against
    realised data, use :func:`stochPathSim.risk.exceedance_counts` instead, which compares the
    realised path with the simulated band.
    """
    simulations = np.asarray(simulations, dtype=float)
    quantiles = {f"Quantile {p}%": np.percentile(simulations, p, axis=0) for p in (99, 90, 10, 1)}
    stats_df = pd.DataFrame({
        "P_LEVEL": np.mean(simulations > initial_value, axis=0),
        "Mean": np.mean(simulations, axis=0),
        **quantiles,
    })
    stats_df.attrs["Upper Exceedance"] = int((initial_value > quantiles["Quantile 99%"]).sum())
    stats_df.attrs["Lower Exceedance"] = int((initial_value < quantiles["Quantile 1%"]).sum())
    return stats_df


def display_statistics(stats_df, time):
    """Print the statistics and exceedance counts; return a copy with a ``Time`` column."""
    out = stats_df.copy()
    out.insert(0, "Time", time)
    print("\nExceedance Statistics:")
    print(f"  Upper Exceedance (Initial > Quantile 99%): {stats_df.attrs['Upper Exceedance']}")
    print(f"  Lower Exceedance (Initial < Quantile 1%): {stats_df.attrs['Lower Exceedance']}")
    print("\nStatistical Summary:")
    print(out)
    return out


def plot_simulation(time, simulations, num_paths=10, title="Simulation Paths",
                    xlabel="Time (Years)", ylabel="Value", ax=None):
    """Plot up to ``num_paths`` simulated paths; return an interactive Plotly figure.

    ``ax`` is accepted for backward compatibility and ignored.
    """
    import plotly.graph_objects as go

    from .viz.theme import new_figure

    fig = new_figure(title, xlabel, ylabel)
    simulations = np.asarray(simulations)
    for i in range(min(int(num_paths), simulations.shape[0])):
        fig.add_trace(go.Scatter(x=time, y=simulations[i], mode="lines", line=dict(width=1),
                                 opacity=0.7, name=f"path {i}", showlegend=False))
    return fig
