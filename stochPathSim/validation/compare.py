"""Side-by-side comparison of models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from ..viz.theme import PALETTE, new_figure, rgba


@dataclass
class ModelComparison:
    """Terminal-distribution statistics of several models and a density overlay figure."""

    table: pd.DataFrame
    figure: go.Figure
    terminal_values: dict


def compare_models(
    models: dict,
    T: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 50_000,
    seed=None,
    state: str | dict | None = None,
    quantiles=(0.01, 0.05, 0.5, 0.95, 0.99),
    bins: int = 120,
    title: str | None = None,
) -> ModelComparison:
    """Simulate several models on the same horizon and compare their terminal distributions.

    Parameters
    ----------
    models : dict
        ``{label: process}``.
    state : str or dict, optional
        State to compare (default: each model's primary state), or ``{label: state}``.

    Returns
    -------
    ModelComparison
        ``table`` has one row per model with mean, std, skewness, excess kurtosis and the requested
        quantiles; ``figure`` overlays the terminal densities.
    """
    rows, values = [], {}
    fig = new_figure(title or f"Terminal distributions at T = {T:g}", "value", "density")
    for i, (label, proc) in enumerate(models.items()):
        st = state.get(label) if isinstance(state, dict) else state
        x = proc.simulate(T=T, n_steps=n_steps, n_paths=n_paths, seed=seed).terminal(st)
        values[label] = x
        row = dict(
            model=label,
            mean=x.mean(),
            std=x.std(ddof=1),
            skew=stats.skew(x),
            excess_kurtosis=stats.kurtosis(x),
        )
        row.update({f"q{q * 100:g}": v for q, v in zip(quantiles, np.quantile(x, quantiles), strict=True)})
        rows.append(row)
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Histogram(
                x=x,
                nbinsx=bins,
                histnorm="probability density",
                name=label,
                marker_color=rgba(color, 0.45),
                opacity=0.9,
            )
        )
    fig.update_layout(barmode="overlay", hovermode="closest", bargap=0)
    return ModelComparison(pd.DataFrame(rows).set_index("model"), fig, values)
