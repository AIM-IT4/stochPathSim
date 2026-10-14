"""Monte Carlo convergence and time-discretisation studies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..pricing.montecarlo import price_mc
from ..viz.theme import ACCENT, PALETTE, template


def _seeds(seed, n):
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n)]


def convergence_study(
    process,
    payoff,
    T: float,
    reference: float,
    n_paths_list=(1_000, 4_000, 16_000, 64_000),
    *,
    n_steps: int = 50,
    seed=None,
    discount=None,
    antithetic: bool = False,
) -> pd.DataFrame:
    """Price with an increasing number of paths and compare with a reference value.

    Every run uses an independent random stream. For an unbiased estimator the error
    shrinks like ``1/sqrt(n_paths)`` and the z-scores ``error / stderr`` stay within a few units.

    Returns a DataFrame with columns ``n_paths, price, stderr, error, abs_error, z_score``;
    ``attrs["reference"]`` holds the reference value.
    """
    rows = []
    for n, s in zip(n_paths_list, _seeds(seed, len(n_paths_list)), strict=True):
        res = price_mc(
            process,
            payoff,
            T,
            discount=discount,
            n_steps=n_steps,
            n_paths=int(n),
            seed=s,
            antithetic=antithetic,
        )
        err = res.price - reference
        rows.append(
            dict(
                n_paths=int(n),
                price=res.price,
                stderr=res.stderr,
                error=err,
                abs_error=abs(err),
                z_score=err / res.stderr if res.stderr > 0 else np.nan,
            )
        )
    df = pd.DataFrame(rows)
    df.attrs["reference"] = float(reference)
    return df


def convergence_plot(df: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Two panels: price with 95% intervals against the reference, and the error on log-log axes
    with the standard error and a ``1/sqrt(N)`` guide."""
    ref = df.attrs.get("reference")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Price and 95% interval", "Error vs paths (log-log)"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=df.n_paths,
            y=df.price,
            mode="markers+lines",
            name="Monte Carlo",
            error_y=dict(type="data", array=1.96 * df.stderr, visible=True),
            line=dict(color=ACCENT),
        ),
        row=1,
        col=1,
    )
    if ref is not None:
        fig.add_hline(
            y=ref, line_dash="dash", line_color=PALETTE[3], annotation_text="reference", row=1, col=1
        )
    fig.add_trace(
        go.Scatter(
            x=df.n_paths, y=df.abs_error, mode="markers+lines", name="|error|", line=dict(color=PALETTE[1])
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df.n_paths,
            y=df.stderr,
            mode="markers+lines",
            name="standard error",
            line=dict(color=ACCENT, dash="dot"),
        ),
        row=1,
        col=2,
    )
    n0, s0 = float(df.n_paths.iloc[0]), float(df.stderr.iloc[0])
    guide = s0 * np.sqrt(n0 / df.n_paths.to_numpy(dtype=float))
    fig.add_trace(
        go.Scatter(
            x=df.n_paths, y=guide, mode="lines", name="∝ 1/√N", line=dict(color=PALETTE[2], dash="dash")
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(type="log", title_text="paths", row=1, col=1)
    fig.update_xaxes(type="log", title_text="paths", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_layout(
        template=template(), title=title or "Monte Carlo convergence", hovermode="closest", height=430
    )
    return fig


def discretisation_study(
    process,
    payoff,
    T: float,
    reference: float,
    n_steps_list=(4, 8, 16, 32, 64),
    *,
    n_paths: int = 100_000,
    seed=None,
    discount=None,
    antithetic: bool = True,
) -> pd.DataFrame:
    """Measure the time-discretisation bias of a scheme.

    ``process`` is a process, or a callable ``n_steps -> process`` for schemes whose parameters
    depend on the step. All runs share the same seed so the statistical noise is similar.

    Returns a DataFrame with columns ``n_steps, dt, price, stderr, bias, abs_bias, significant``
    (``|bias| > 3 stderr``). ``attrs["weak_order"]`` is the log-log slope of ``|bias|`` against
    ``dt`` over the significant points (NaN if fewer than two), an estimate of the weak order.
    """
    rows = []
    for n in n_steps_list:
        proc = process(int(n)) if callable(process) and not hasattr(process, "simulate") else process
        res = price_mc(
            proc,
            payoff,
            T,
            discount=discount,
            n_steps=int(n),
            n_paths=int(n_paths),
            seed=seed,
            antithetic=antithetic,
        )
        bias = res.price - reference
        rows.append(
            dict(
                n_steps=int(n),
                dt=T / int(n),
                price=res.price,
                stderr=res.stderr,
                bias=bias,
                abs_bias=abs(bias),
                significant=bool(abs(bias) > 3 * res.stderr),
            )
        )
    df = pd.DataFrame(rows)
    sig = df[df.significant & (df.abs_bias > 0)]
    order = float(np.polyfit(np.log(sig.dt), np.log(sig.abs_bias), 1)[0]) if len(sig) >= 2 else float("nan")
    df.attrs["weak_order"] = order
    df.attrs["reference"] = float(reference)
    return df


def discretisation_plot(df: pd.DataFrame, title: str | None = None) -> go.Figure:
    """|bias| against the step size on log-log axes, with the 3-standard-error noise floor."""
    order = df.attrs.get("weak_order", float("nan"))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.dt,
            y=df.abs_bias,
            mode="markers+lines",
            name="|bias|",
            line=dict(color=ACCENT, width=3),
            marker=dict(size=9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.dt,
            y=3 * df.stderr,
            mode="lines",
            name="3 × standard error",
            line=dict(color=PALETTE[3], dash="dash"),
        )
    )
    label = f"weak order ≈ {order:.2f}" if np.isfinite(order) else "bias below noise"
    fig.update_layout(
        template=template(),
        title=title or f"Discretisation bias ({label})",
        xaxis=dict(type="log", title="time step dt (years)"),
        yaxis=dict(type="log", title="|price − reference|"),
        hovermode="closest",
    )
    return fig
