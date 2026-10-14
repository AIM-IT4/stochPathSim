"""Interactive risk dashboards: exposure, backtests and VaR."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from .theme import ACCENT, PALETTE, new_figure, rgba, template


def exposure_dashboard(profile, title: str | None = None) -> go.Figure:
    """EE, ENE, PFE and Effective EE of an :class:`~stochPathSim.risk.ExposureProfile`, with EPE marked."""
    t = profile.times
    fig = new_figure(title or "Exposure profile", "time (years)", "exposure")
    fig.add_trace(
        go.Scatter(
            x=t,
            y=profile.pfe,
            mode="lines",
            name=f"PFE {profile.pfe_level:.0%}",
            line=dict(width=3, color=PALETTE[3]),
            fill="tozeroy",
            fillcolor=rgba(PALETTE[3], 0.08),
        )
    )
    fig.add_trace(go.Scatter(x=t, y=profile.ee, mode="lines", name="EE", line=dict(width=3, color=ACCENT)))
    fig.add_trace(
        go.Scatter(
            x=t,
            y=profile.effective_ee,
            mode="lines",
            name="Effective EE",
            line=dict(width=2, dash="dash", color=PALETTE[4]),
        )
    )
    fig.add_trace(
        go.Scatter(x=t, y=profile.ene, mode="lines", name="ENE", line=dict(width=2, color=PALETTE[2]))
    )
    fig.add_hline(
        y=profile.epe,
        line_dash="dot",
        line_color=PALETTE[1],
        annotation_text=f"EPE {profile.epe:,.4g}",
        annotation_position="top left",
    )
    fig.update_traces(hovertemplate="%{y:,.4g}")
    return fig


def backtest_dashboard(
    pit, realised=None, bands=None, level: float = 0.99, times=None, title: str | None = None
) -> go.Figure:
    """Three-panel risk-factor backtest.

    1. histogram of PIT values against the uniform density;
    2. QQ plot of sorted PIT values against uniform quantiles;
    3. exceedance timeline: realised values against the simulated ``bands`` (``(lower, upper)``
       arrays), or PIT values against ``1 - level`` / ``level`` thresholds when no bands are given.
    """
    pit = np.asarray(pit, dtype=float)
    n = pit.size
    x = np.arange(n) if times is None else np.asarray(times)
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("PIT histogram", "PIT QQ plot", "Exceedance timeline"),
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )
    fig.add_trace(
        go.Histogram(
            x=pit,
            xbins=dict(start=0, end=1, size=0.05),
            histnorm="probability density",
            marker_color=rgba(ACCENT, 0.7),
            name="PIT",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[1, 1], mode="lines", name="uniform", line=dict(color=PALETTE[3], dash="dash")
        ),
        row=1,
        col=1,
    )
    u = (np.arange(1, n + 1) - 0.5) / n
    fig.add_trace(
        go.Scatter(
            x=u, y=np.sort(pit), mode="markers", name="PIT quantiles", marker=dict(size=4, color=ACCENT)
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="45° line",
            line=dict(color=PALETTE[3], dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    if realised is not None and bands is not None:
        lo, hi = (np.asarray(b, dtype=float) for b in bands)
        r = np.asarray(realised, dtype=float)
        fig.add_trace(
            go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lo,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=rgba(ACCENT, 0.15),
                name="simulated band",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=r, mode="lines", name="realised", line=dict(color=PALETTE[1])), row=2, col=1
        )
        out = (r > hi) | (r < lo)
        y_out = r[out]
    else:
        fig.add_trace(
            go.Scatter(x=x, y=pit, mode="markers", name="PIT", marker=dict(size=4, color=ACCENT)),
            row=2,
            col=1,
        )
        for thr in (1 - level, level):
            fig.add_hline(y=thr, line_dash="dash", line_color=PALETTE[3], row=2, col=1)
        out = (pit > level) | (pit < 1 - level)
        y_out = pit[out]
    fig.add_trace(
        go.Scatter(
            x=x[out],
            y=y_out,
            mode="markers",
            name=f"exceedances ({int(out.sum())})",
            marker=dict(size=9, color=PALETTE[3], symbol="x"),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        template=template(),
        title=title or "Risk-factor backtest",
        height=720,
        hovermode="closest",
        bargap=0.03,
    )
    fig.update_xaxes(title_text="PIT", row=1, col=1)
    fig.update_xaxes(title_text="uniform quantile", row=1, col=2)
    fig.update_yaxes(title_text="PIT quantile", row=1, col=2)
    return fig


def var_plot(pnl, level: float = 0.99, bins: int = 80, title: str | None = None) -> go.Figure:
    """P&L histogram with historical VaR and Expected Shortfall marked (losses are positive)."""
    pnl = np.asarray(pnl, dtype=float)
    var = -np.quantile(pnl, 1 - level)
    es = -pnl[pnl <= -var].mean()
    fig = new_figure(title or f"P&L distribution, {level:.0%} VaR and ES", "P&L", "density")
    fig.add_trace(
        go.Histogram(
            x=pnl, nbinsx=bins, histnorm="probability density", name="P&L", marker_color=rgba(ACCENT, 0.65)
        )
    )
    grid = np.linspace(pnl.min(), pnl.max(), 300)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=stats.norm.pdf(grid, pnl.mean(), pnl.std(ddof=1)),
            mode="lines",
            name="normal fit",
            line=dict(color=PALETTE[4], dash="dot"),
        )
    )
    fig.add_vline(x=-var, line_color=PALETTE[3], line_width=2, annotation_text=f"VaR {var:,.4g}")
    fig.add_vline(
        x=-es,
        line_color=PALETTE[3],
        line_dash="dash",
        annotation_text=f"ES {es:,.4g}",
        annotation_position="bottom left",
    )
    fig.update_layout(hovermode="closest", bargap=0.02)
    return fig
