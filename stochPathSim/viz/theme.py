"""The stochPathSim Plotly theme."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

PALETTE = [
    "#3B82F6",  # blue
    "#F59E0B",  # amber
    "#10B981",  # emerald
    "#EF4444",  # red
    "#8B5CF6",  # violet
    "#06B6D4",  # cyan
    "#EC4899",  # pink
    "#84CC16",  # lime
]
ACCENT = PALETTE[0]
FONT = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"

_STYLES = {
    "light": dict(bg="#FFFFFF", plot="#FFFFFF", grid="#E5E7EB", text="#111827", muted="#6B7280"),
    "dark": dict(bg="#0F172A", plot="#0F172A", grid="#1E293B", text="#E2E8F0", muted="#94A3B8"),
}


def _template(style: str) -> go.layout.Template:
    s = _STYLES[style]
    axis = dict(
        gridcolor=s["grid"],
        zerolinecolor=s["grid"],
        linecolor=s["grid"],
        tickcolor=s["grid"],
        title=dict(font=dict(color=s["muted"])),
        tickfont=dict(color=s["muted"]),
    )
    return go.layout.Template(
        layout=dict(
            colorway=PALETTE,
            font=dict(family=FONT, size=13, color=s["text"]),
            title=dict(font=dict(size=17, color=s["text"]), x=0.02, xanchor="left"),
            paper_bgcolor=s["bg"],
            plot_bgcolor=s["plot"],
            xaxis=axis,
            yaxis=axis,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=s["text"])),
            hoverlabel=dict(font=dict(family=FONT)),
            hovermode="x unified",
            margin=dict(l=60, r=24, t=60, b=50),
            scene=dict(xaxis=axis, yaxis=axis, zaxis=axis),
        )
    )


pio.templates["stochpathsim"] = _template("light")
pio.templates["stochpathsim_dark"] = _template("dark")
_current = {"name": "stochpathsim"}


def set_theme(style: str = "light") -> str:
    """Use the light or dark stochPathSim theme for every figure the package creates.

    Returns the name of the Plotly template now in use.
    """
    if style not in _STYLES:
        raise ValueError("style must be 'light' or 'dark'")
    _current["name"] = "stochpathsim" if style == "light" else "stochpathsim_dark"
    return _current["name"]


def template() -> str:
    """Name of the Plotly template currently used by stochPathSim figures."""
    return _current["name"]


def rgba(hex_color: str, alpha: float) -> str:
    """``#RRGGBB`` to an ``rgba(...)`` string."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def new_figure(title: str | None = None, xaxis: str | None = None, yaxis: str | None = None) -> go.Figure:
    """An empty figure with the stochPathSim theme applied."""
    fig = go.Figure()
    fig.update_layout(template=template(), title=title, xaxis_title=xaxis, yaxis_title=yaxis)
    return fig
