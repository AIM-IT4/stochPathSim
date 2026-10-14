"""Interactive charts of simulated paths and their distributions."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .theme import ACCENT, PALETTE, new_figure, rgba, template


def _label(result, name):
    return name or result.primary


def _band(fig, t, lo, hi, color, alpha, label, group=None, visible=True):
    fig.add_trace(
        go.Scatter(
            x=t,
            y=hi,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=group,
            visible=visible,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=lo,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=rgba(color, alpha),
            name=label,
            legendgroup=group,
            hovertemplate=f"{label}<extra></extra>",
            visible=visible,
        )
    )


def paths(
    result,
    name: str | None = None,
    n_paths: int = 30,
    quantiles=(0.01, 0.99),
    show_mean: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Sample paths with the mean and a quantile band.

    Traces: the quantile band (2 traces, if ``quantiles``), one trace per sample path, and the mean.
    """
    label = _label(result, name)
    p = result.get(name)
    t = result.times
    fig = new_figure(title or f"{label}: {p.shape[0]:,} simulated paths", "time (years)", label)
    if quantiles:
        lo, hi = np.quantile(p, quantiles, axis=0)
        _band(fig, t, lo, hi, ACCENT, 0.15, f"{quantiles[0]:.0%}–{quantiles[1]:.0%} band")
    for i in range(min(int(n_paths), p.shape[0])):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=p[i],
                mode="lines",
                line=dict(width=1),
                opacity=0.55,
                showlegend=False,
                hoverinfo="skip",
                legendgroup="paths",
                name=f"path {i}",
            )
        )
    if show_mean:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=p.mean(axis=0),
                mode="lines",
                name="mean",
                line=dict(width=3, color="#111827" if template() == "stochpathsim" else "#F8FAFC"),
            )
        )
    fig.update_layout(hovermode="x")
    return fig


def fan(
    result, name: str | None = None, quantiles=(0.01, 0.05, 0.25, 0.75, 0.95, 0.99), title: str | None = None
) -> go.Figure:
    """Fan chart: nested symmetric quantile bands around the median."""
    label = _label(result, name)
    p = result.get(name)
    t = result.times
    qs = sorted(quantiles)
    values = np.quantile(p, qs, axis=0)
    fig = new_figure(title or f"{label}: distribution over time", "time (years)", label)
    n = len(qs) // 2
    for i in range(n):
        _band(
            fig,
            t,
            values[i],
            values[-1 - i],
            ACCENT,
            0.12 + 0.5 * i / max(n, 1),
            f"{qs[i]:.0%}–{qs[-1 - i]:.0%}",
        )
    fig.add_trace(
        go.Scatter(x=t, y=np.median(p, axis=0), mode="lines", name="median", line=dict(width=3, color=ACCENT))
    )
    return fig


def path_explorer(result, max_paths: int = 50, title: str | None = None) -> go.Figure:
    """Explore every state of a result: a dropdown picks the state, a slider the number of paths."""
    t = result.times
    names = result.names
    k = min(int(max_paths), result.n_paths)
    fig = new_figure(title or "Path explorer", "time (years)", None)
    groups = []
    for j, nm in enumerate(names):
        p = result.get(nm)
        first = len(fig.data)
        for i in range(k):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=p[i],
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.55,
                    showlegend=False,
                    hoverinfo="skip",
                    visible=j == 0,
                    name=f"{nm} path {i}",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=p.mean(axis=0),
                mode="lines",
                name=f"{nm} mean",
                line=dict(width=3, color=PALETTE[j % len(PALETTE)]),
                visible=j == 0,
            )
        )
        groups.append(range(first, len(fig.data)))
    n_traces = len(fig.data)
    buttons = []
    for j, nm in enumerate(names):
        vis = [i in groups[j] for i in range(n_traces)]
        buttons.append(dict(label=nm, method="update", args=[{"visible": vis}, {"yaxis.title.text": nm}]))
    counts = sorted({c for c in (1, 5, 10, 25, k) if c <= k})
    steps = []
    for c in counts:
        opacity = []
        for g in groups:
            opacity += [0.55 if (idx - g.start) < c else 0.0 for idx in g][:-1] + [1.0]
        steps.append(dict(label=str(c), method="restyle", args=[{"opacity": opacity}]))
    fig.update_layout(
        yaxis_title=names[0],
        hovermode="x",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=1.0,
                xanchor="right",
                y=1.15,
                yanchor="top",
                showactive=True,
            )
        ],
        sliders=[
            dict(
                active=len(counts) - 1, steps=steps, currentvalue=dict(prefix="paths shown: "), pad=dict(t=40)
            )
        ],
    )
    return fig


def terminal_distribution(
    result,
    name: str | None = None,
    t: float | None = None,
    bins: int = 80,
    pdf=None,
    title: str | None = None,
) -> go.Figure:
    """Histogram of values at time ``t`` (default: the horizon), with an optional analytic density.

    ``pdf`` is a callable evaluated on a grid of values and overlaid as a line.
    """
    label = _label(result, name)
    values = result.terminal(name) if t is None else result.at(t, name)
    when = result.times[-1] if t is None else t
    fig = new_figure(title or f"{label} at t = {when:g}", label, "density")
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            histnorm="probability density",
            name="simulated",
            marker=dict(color=rgba(ACCENT, 0.65), line=dict(width=0)),
        )
    )
    if pdf is not None:
        grid = np.linspace(np.quantile(values, 0.0005), np.quantile(values, 0.9995), 400)
        fig.add_trace(
            go.Scatter(
                x=grid, y=pdf(grid), mode="lines", name="analytic", line=dict(width=3, color=PALETTE[3])
            )
        )
    fig.add_vline(
        x=float(np.mean(values)),
        line_dash="dash",
        line_color=PALETTE[1],
        annotation_text="mean",
        annotation_position="top",
    )
    fig.update_layout(bargap=0.02, hovermode="closest")
    return fig


def _density_grid(p, n_bins, n_times):
    idx = np.unique(np.linspace(1, p.shape[1] - 1, min(n_times, p.shape[1] - 1)).astype(int))
    lo, hi = np.quantile(p[:, idx], [0.001, 0.999])
    edges = np.linspace(lo, hi, n_bins + 1)
    dens = np.stack([np.histogram(p[:, i], bins=edges, density=True)[0] for i in idx])
    return idx, 0.5 * (edges[:-1] + edges[1:]), dens


def density_evolution(
    result,
    name: str | None = None,
    kind: str = "heatmap",
    n_bins: int = 60,
    n_times: int = 60,
    title: str | None = None,
) -> go.Figure:
    """How the cross-sectional distribution evolves over time.

    ``kind`` is ``"heatmap"`` (time x value), ``"surface"`` (3-D density surface) or
    ``"animation"`` (a histogram per time step, played with a slider).
    """
    label = _label(result, name)
    p = result.get(name)
    idx, centres, dens = _density_grid(p, n_bins, n_times)
    times = result.times[idx]
    title = title or f"{label}: density over time"
    if kind == "heatmap":
        fig = new_figure(title, "time (years)", label)
        fig.add_trace(
            go.Heatmap(x=times, y=centres, z=dens.T, colorscale="Viridis", colorbar=dict(title="density"))
        )
        fig.update_layout(hovermode="closest")
        return fig
    if kind == "surface":
        fig = new_figure(title)
        fig.add_trace(
            go.Surface(x=times, y=centres, z=dens.T, colorscale="Viridis", colorbar=dict(title="density"))
        )
        fig.update_layout(
            scene=dict(xaxis_title="time (years)", yaxis_title=label, zaxis_title="density"), height=600
        )
        return fig
    if kind == "animation":
        fig = new_figure(title, label, "density")
        fig.add_trace(go.Bar(x=centres, y=dens[0], marker_color=ACCENT, name=f"t = {times[0]:.3g}"))
        fig.frames = [
            go.Frame(data=[go.Bar(x=centres, y=d, marker_color=ACCENT)], name=f"{tt:.4g}")
            for tt, d in zip(times, dens, strict=True)
        ]
        fig.update_layout(
            yaxis_range=[0, float(dens.max()) * 1.05],
            bargap=0,
            hovermode="closest",
            updatemenus=[
                dict(
                    type="buttons",
                    x=0.0,
                    y=1.15,
                    xanchor="left",
                    buttons=[
                        dict(
                            label="▶ play",
                            method="animate",
                            args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)],
                        ),
                        dict(
                            label="❚❚ pause",
                            method="animate",
                            args=[[None], dict(mode="immediate", frame=dict(duration=0))],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    currentvalue=dict(prefix="t = "),
                    pad=dict(t=40),
                    steps=[
                        dict(
                            label=f"{tt:.3g}",
                            method="animate",
                            args=[[f"{tt:.4g}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                        )
                        for tt in times
                    ],
                )
            ],
        )
        return fig
    raise ValueError("kind must be 'heatmap', 'surface' or 'animation'")


def correlation_heatmap(result, on: str = "increments", names=None, title: str | None = None) -> go.Figure:
    """Correlation between the states of a result.

    ``on="increments"`` correlates step increments pooled over paths and time (the
    instantaneous correlation); ``on="terminal"`` correlates the terminal values.
    """
    names = list(names or result.names)
    if len(names) < 2:
        raise ValueError("a correlation heatmap needs at least two states")
    if on == "increments":
        data = np.stack([np.diff(result.get(n), axis=1).ravel() for n in names])
    elif on == "terminal":
        data = np.stack([result.terminal(n) for n in names])
    else:
        raise ValueError("on must be 'increments' or 'terminal'")
    corr = np.corrcoef(data)
    fig = new_figure(title or f"Correlation of {on}")
    fig.add_trace(
        go.Heatmap(
            x=names,
            y=names,
            z=corr,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=np.round(corr, 3),
            texttemplate="%{text}",
            colorbar=dict(title="ρ"),
        )
    )
    fig.update_layout(yaxis_autorange="reversed", hovermode="closest")
    return fig
