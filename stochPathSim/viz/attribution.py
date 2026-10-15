"""Plotly risk-contribution charts for normal portfolio VaR and ES."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .theme import template


def portfolio_risk_chart(risk, labels=None):
    """Interactive stacked comparison of signed component VaR and ES.

    Parameters
    ----------
    risk : PortfolioRisk
        Output of stochPathSim.risk.normal_portfolio_risk.
    labels : sequence of str, optional
        Labels for each asset; defaults to asset_0, asset_1, etc.

    Returns
    -------
    plotly.graph_objects.Figure
        Grouped bars per asset, retaining signed contributions.
    """
    n = len(risk.component_var)
    if labels is None:
        labels = [f"asset_{i}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length must equal the number of assets")
    if not np.all(np.isfinite(risk.component_var)) or not np.all(np.isfinite(risk.component_es)):
        raise ValueError("risk contributions must be finite")
    fig = go.Figure()
    fig.add_bar(x=list(labels), y=risk.component_var, name=f"VaR ({risk.level:.1%})")
    fig.add_bar(x=list(labels), y=risk.component_es, name=f"ES ({risk.level:.1%})")
    fig.update_layout(
        title="Portfolio risk contributions",
        template=template(),
        barmode="group",
        xaxis_title="Position",
        yaxis_title="Contribution to portfolio loss",
    )
    return fig
