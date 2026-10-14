"""Interactive charts for interest-rate models."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .paths import fan
from .theme import ACCENT, PALETTE, new_figure, rgba


def curve_fit_check(result, curve, title: str | None = None) -> go.Figure:
    """Monte Carlo discount factors (with a 95% band) against the initial curve ``P(0, t)``.

    ``result`` must contain a ``discount`` state, as every short-rate model does.
    """
    d = result["discount"]
    t = result.times
    mean = d.mean(axis=0)
    se = d.std(axis=0, ddof=1) / np.sqrt(d.shape[0])
    fig = new_figure(title or "Curve fit: Monte Carlo vs initial curve", "maturity (years)", "P(0, t)")
    fig.add_trace(
        go.Scatter(
            x=t, y=mean + 1.96 * se, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=mean - 1.96 * se,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=rgba(ACCENT, 0.2),
            name="95% MC band",
        )
    )
    fig.add_trace(go.Scatter(x=t, y=mean, mode="lines", name="Monte Carlo", line=dict(width=3, color=ACCENT)))
    fig.add_trace(
        go.Scatter(
            x=t,
            y=curve.discount(t),
            mode="lines",
            name="curve",
            line=dict(width=2, dash="dash", color=PALETTE[3]),
        )
    )
    return fig


def short_rate_fan(result, quantiles=(0.01, 0.05, 0.25, 0.75, 0.95, 0.99), title: str | None = None):
    """Fan chart of the simulated short rate."""
    return fan(result, "rate", quantiles=quantiles, title=title or "Short rate: distribution over time")
