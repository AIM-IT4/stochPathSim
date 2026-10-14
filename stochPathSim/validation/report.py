"""Self-contained HTML model-validation reports."""

from __future__ import annotations

import html
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .._version import __version__
from ..risk.backtest import (
    anderson_darling_uniform,
    christoffersen_independence,
    kupiec_pof,
    pit_values,
)
from ..viz.risk import backtest_dashboard
from .compare import compare_models
from .convergence import convergence_plot, convergence_study, discretisation_plot, discretisation_study


@dataclass
class Check:
    """One pass/fail test in a report."""

    section: str
    name: str
    value: float
    threshold: str
    passed: bool | None  # None: informational


@dataclass
class _Section:
    title: str
    description: str
    tables: list = field(default_factory=list)
    figures: list = field(default_factory=list)
    model: str = ""


_CSS = """
:root { --bg:#f8fafc; --card:#fff; --text:#0f172a; --muted:#64748b; --line:#e2e8f0;
        --accent:#3b82f6; --pass:#059669; --fail:#dc2626; --info:#64748b; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font-family: Inter, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height:1.5; }
header { background:linear-gradient(120deg,#0f172a,#1e3a8a); color:#fff; padding:36px 48px; }
header h1 { margin:0 0 6px; font-size:28px; letter-spacing:-0.02em; }
header .meta { color:#cbd5e1; font-size:13px; }
main { max-width:1200px; margin:0 auto; padding:28px 32px 64px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:24px 28px;
        margin-bottom:24px; box-shadow:0 1px 2px rgba(15,23,42,.04); }
h2 { margin:0 0 4px; font-size:20px; }
.desc { color:var(--muted); margin:0 0 16px; }
.model { font-family: ui-monospace, Consolas, monospace; font-size:12px; color:var(--muted);
         background:#f1f5f9; padding:6px 10px; border-radius:8px; display:inline-block; margin-bottom:14px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; margin:8px 0 18px; }
th, td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:right; }
th { color:var(--muted); font-weight:600; background:#f8fafc; }
td:first-child, th:first-child { text-align:left; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:700;
         color:#fff; }
.pass { background:var(--pass); } .fail { background:var(--fail); } .info { background:var(--info); }
.kpis { display:flex; gap:16px; flex-wrap:wrap; margin-top:4px; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:12px;
       padding:14px 20px; min-width:150px; }
.kpi b { font-size:26px; display:block; } .kpi span { color:var(--muted); font-size:13px; }
footer { color:var(--muted); font-size:12px; text-align:center; padding:24px; }
"""


def _fmt(x):
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "–"
        return f"{x:,.6g}"
    return html.escape(str(x))


def _table(df: pd.DataFrame, index: bool = False) -> str:
    cols = ([df.index.name or ""] if index else []) + list(df.columns)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = []
    for idx, row in df.iterrows():
        cells = ([f"<td>{_fmt(idx)}</td>"] if index else []) + [f"<td>{_fmt(v)}</td>" for v in row]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _badge(passed):
    if passed is None:
        return '<span class="badge info">INFO</span>'
    return '<span class="badge pass">PASS</span>' if passed else '<span class="badge fail">FAIL</span>'


class ValidationReport:
    """Collects validation studies and writes them to one self-contained HTML file.

    >>> rep = ValidationReport("Heston model validation")
    >>> rep.add_convergence(model, EuropeanCall(100), 1.0, reference=exact, discount=0.03)  # doctest: +SKIP
    >>> rep.to_html("heston_validation.html")  # doctest: +SKIP

    Parameters
    ----------
    title : str
        Report title.
    include_plotlyjs : {"inline", "cdn"}
        ``"inline"`` embeds plotly.js once so the file works offline; ``"cdn"`` loads it from the web.
    z_threshold : float
        Monte Carlo prices pass when ``|z| = |error| / stderr`` is below this.
    p_threshold : float
        Backtest tests pass when their p-value is above this.
    """

    def __init__(
        self,
        title: str = "Model validation report",
        include_plotlyjs: str = "inline",
        z_threshold: float = 3.0,
        p_threshold: float = 0.05,
    ):
        if include_plotlyjs not in ("inline", "cdn"):
            raise ValueError("include_plotlyjs must be 'inline' or 'cdn'")
        self.title = title
        self.include_plotlyjs = include_plotlyjs
        self.z_threshold = z_threshold
        self.p_threshold = p_threshold
        self.sections: list[_Section] = []
        self.checks: list[Check] = []

    # ------------------------------------------------------------------ sections
    def add_convergence(
        self, process, payoff, T: float, reference: float, *, label: str | None = None, **kwargs
    ) -> pd.DataFrame:
        """Monte Carlo convergence against a reference price (see :func:`convergence_study`)."""
        title = label or "Monte Carlo convergence"
        df = convergence_study(process, payoff, T, reference, **kwargs)
        worst = float(np.nanmax(np.abs(df.z_score)))
        self.checks.append(
            Check(
                title,
                "max |z-score| across path counts",
                worst,
                f"< {self.z_threshold:g}",
                worst < self.z_threshold,
            )
        )
        sec = _Section(
            title,
            f"Prices with an increasing number of paths against the reference "
            f"{reference:.6g}. An unbiased estimator stays within a few standard errors "
            f"and its error falls like 1/√N.",
            model=repr(process),
        )
        sec.tables.append(df)
        sec.figures.append(convergence_plot(df, title=title))
        self.sections.append(sec)
        return df

    def add_discretisation(
        self, process, payoff, T: float, reference: float, *, label: str | None = None, **kwargs
    ) -> pd.DataFrame:
        """Time-discretisation bias against a reference price (see :func:`discretisation_study`)."""
        title = label or "Time discretisation"
        df = discretisation_study(process, payoff, T, reference, **kwargs)
        finest = df.iloc[-1]
        z = float(finest.abs_bias / finest.stderr) if finest.stderr > 0 else float("nan")
        self.checks.append(
            Check(
                title,
                f"|bias| / stderr at {int(finest.n_steps)} steps",
                z,
                f"< {self.z_threshold:g}",
                z < self.z_threshold,
            )
        )
        self.checks.append(Check(title, "estimated weak order", df.attrs["weak_order"], "reported", None))
        sec = _Section(
            title,
            "Price against the number of time steps on the same random stream. The slope "
            "of |bias| against dt on log-log axes estimates the weak order of the scheme.",
            model=repr(process) if hasattr(process, "simulate") else getattr(process, "__name__", ""),
        )
        sec.tables.append(df)
        sec.figures.append(discretisation_plot(df, title=title))
        self.sections.append(sec)
        return df

    def add_comparison(self, models: dict, T: float = 1.0, *, label: str | None = None, **kwargs):
        """Terminal-distribution comparison of several models (see :func:`compare_models`)."""
        title = label or "Model comparison"
        comp = compare_models(models, T=T, title=title, **kwargs)
        sec = _Section(
            title,
            "Moments and quantiles of the terminal distribution of each model on the same "
            "horizon: how much skew and tail each model generates.",
        )
        sec.tables.append(comp.table)
        sec.figures.append(comp.figure)
        self.sections.append(sec)
        self.checks.append(Check(title, "models compared", float(len(models)), "–", None))
        return comp

    def add_backtest(
        self, simulated_distributions, realised, level: float = 0.99, *, label: str | None = None, times=None
    ) -> dict:
        """Backtest realised values against simulated distributions (one row per date).

        Runs the Anderson-Darling uniformity test on the PIT values, the Kupiec test on two-sided
        exceedances of the ``[1 - level, level]`` quantile band, and Christoffersen's independence test.
        """
        title = label or "Risk-factor backtest"
        pit = pit_values(simulated_distributions, realised)
        exceed = (pit > level) | (pit < 1 - level)
        n, k = pit.size, int(exceed.sum())
        ad, p_ad = anderson_darling_uniform(pit)
        two_sided = 1 - 2 * (1 - level)
        lr_k, p_k = kupiec_pof(n, k, level=two_sided)
        lr_c, p_c = christoffersen_independence(exceed.astype(int))
        out = dict(
            n_obs=n,
            exceedances=k,
            expected=n * 2 * (1 - level),
            ad_stat=ad,
            ad_pvalue=p_ad,
            kupiec_lr=lr_k,
            kupiec_pvalue=p_k,
            christoffersen_lr=lr_c,
            christoffersen_pvalue=p_c,
        )
        thr = f"p > {self.p_threshold:g}"
        self.checks += [
            Check(title, "Anderson-Darling (PIT uniform)", p_ad, thr, p_ad > self.p_threshold),
            Check(
                title,
                f"Kupiec POF ({k} of {n} outside {1 - level:.0%}–{level:.0%})",
                p_k,
                thr,
                p_k > self.p_threshold,
            ),
            Check(
                title, "Christoffersen independence", p_c, thr, bool(np.isnan(p_c) or p_c > self.p_threshold)
            ),
        ]
        sec = _Section(
            title,
            "Probability integral transform (PIT) of each realised value under its "
            "simulated distribution. A correct model gives uniform, independent PITs.",
        )
        sec.tables.append(pd.DataFrame([out]))
        sec.figures.append(backtest_dashboard(pit, level=level, times=times, title=title))
        self.sections.append(sec)
        return out

    def add_figure(self, figure, title: str, description: str = "") -> None:
        """Add any Plotly figure as its own section."""
        sec = _Section(title, description)
        sec.figures.append(figure)
        self.sections.append(sec)

    # ------------------------------------------------------------------ output
    def summary(self) -> pd.DataFrame:
        """One row per check: section, check, value, threshold and result."""
        return pd.DataFrame(
            [
                dict(
                    section=c.section,
                    check=c.name,
                    value=c.value,
                    threshold=c.threshold,
                    result="INFO" if c.passed is None else ("PASS" if c.passed else "FAIL"),
                )
                for c in self.checks
            ]
        )

    @property
    def passed(self) -> bool:
        """True when no check failed."""
        return all(c.passed is not False for c in self.checks)

    def to_html(self, path) -> Path:
        """Write the report and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        js_mode = True if self.include_plotlyjs == "inline" else "cdn"
        first = [True]

        def fig_html(fig):
            out = fig.to_html(
                full_html=False,
                include_plotlyjs=js_mode if first[0] else False,
                config={"displaylogo": False, "responsive": True},
            )
            first[0] = False
            return out

        n_pass = sum(c.passed is True for c in self.checks)
        n_fail = sum(c.passed is False for c in self.checks)
        rows = "".join(
            f"<tr><td>{html.escape(c.section)}</td><td>{html.escape(c.name)}</td><td>{_fmt(c.value)}</td>"
            f"<td>{html.escape(c.threshold)}</td><td>{_badge(c.passed)}</td></tr>"
            for c in self.checks
        )
        summary = (
            f"<table><thead><tr><th>Section</th><th>Check</th><th>Value</th><th>Threshold</th>"
            f"<th>Result</th></tr></thead><tbody>{rows}</tbody></table>"
        )
        body = []
        for sec in self.sections:
            parts = [
                f"<h2>{html.escape(sec.title)}</h2>",
                f'<p class="desc">{html.escape(sec.description)}</p>',
            ]
            if sec.model:
                parts.append(f'<div class="model">{html.escape(sec.model)}</div>')
            parts += [_table(t, index=not isinstance(t.index, pd.RangeIndex)) for t in sec.tables]
            parts += [fig_html(f) for f in sec.figures]
            body.append(f'<section class="card">{"".join(parts)}</section>')
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        overall = _badge(self.passed if self.checks else None)
        doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(self.title)}</title><style>{_CSS}</style></head>
<body><header><h1>{html.escape(self.title)}</h1>
<div class="meta">stochPathSim {__version__} · Python {platform.python_version()} · NumPy {np.__version__}
 · generated {stamp}</div></header>
<main><section class="card"><h2>Summary {overall}</h2>
<div class="kpis"><div class="kpi"><b>{n_pass}</b><span>checks passed</span></div>
<div class="kpi"><b>{n_fail}</b><span>checks failed</span></div>
<div class="kpi"><b>{len(self.sections)}</b><span>sections</span></div></div>
{summary}</section>
{"".join(body)}</main>
<footer>Generated with stochPathSim · https://github.com/AIM-IT4/stochPathSim</footer></body></html>"""
        path.write_text(doc, encoding="utf-8")
        return path

    def __repr__(self) -> str:
        return f"ValidationReport({self.title!r}, sections={len(self.sections)}, checks={len(self.checks)})"
