"""Negative-rate forward option and CIR++ yield-curve fitting demonstration.

Run: python examples/07_displaced_and_cirpp.py
Writes a self-contained interactive Plotly figure in examples/output/.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import stochPathSim as sps
from stochPathSim.analytics import displaced_diffusion_price
from stochPathSim.pricing import EuropeanCall, price_mc

# A forward can be negative, but F + displacement remains strictly positive.
F, K, shift, sigma, T, r = -0.015, -0.01, 0.06, 0.4, 1.0, 0.03
forward = sps.DisplacedDiffusion(s0=F, mu=0.0, sigma=sigma, shift=shift)
mc = price_mc(forward, EuropeanCall(K), T, discount=r,
              n_paths=100_000, n_steps=1, seed=42, antithetic=True)
closed = displaced_diffusion_price(F, K, T, np.exp(-r * T), sigma, shift)
print(f"Displaced forward call: MC {mc.price:.6f} +/- {mc.stderr:.6f}; closed {closed:.6f}")

# CIR++ shifts a positive CIR factor to reproduce the initial market curve.
curve = sps.ZeroCurve([0.25, 1.0, 3.0], [0.02, 0.026, 0.035])
rates = sps.CIRPlusPlus(x0=0.03, kappa=1.2, theta=0.035, sigma=0.14, curve=curve)
result = rates.simulate(T=3, n_steps=120, n_paths=10_000, seed=42)
print(f"CIR++ 3y bond: MC {result['discount'][:, -1].mean():.6f}; market {curve.discount(3):.6f}")
print("Pathwise CIR++ discounts use a trapezoidal integral of the CIR factor.")

short = forward.simulate(T=1, n_steps=60, n_paths=30, seed=9)
fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "Displaced forward paths (negative values allowed)",
    "CIR++ zero-curve check (Monte Carlo vs market)",
))
for idx, path in enumerate(short["spot"][:20]):
    fig.add_trace(go.Scatter(x=short.times, y=path, mode="lines", opacity=0.35,
                             line=dict(color="#3465A4", width=1.2),
                             showlegend=idx == 0, name="Forward paths"), row=1, col=1)
fig.add_trace(go.Scatter(x=result.times[1:], y=result["discount"][:, 1:].mean(axis=0),
                         name="CIR++ Monte Carlo", mode="lines",
                         line=dict(color="#0F766E")), row=1, col=2)
fig.add_trace(go.Scatter(x=result.times[1:], y=curve.discount(result.times[1:]),
                         name="Market discount curve", mode="lines",
                         line=dict(color="#D97706", dash="dash")), row=1, col=2)
fig.update_layout(template="plotly_white", height=490,
                  title="stochPathSim: displaced forwards and curve-fitted CIR++")
fig.update_xaxes(title_text="Years")
fig.update_yaxes(title_text="Forward", row=1, col=1)
fig.update_yaxes(title_text="Discount factor", row=1, col=2)
output = Path(__file__).parent / "output" / "displaced-and-cirpp.html"
output.parent.mkdir(exist_ok=True)
fig.write_html(output, include_plotlyjs=True)
print(f"Interactive Plotly demo: {output}")
