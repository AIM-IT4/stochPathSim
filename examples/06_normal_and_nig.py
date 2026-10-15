"""Normal forwards and heavy-tailed NIG equities, with a self-contained Plotly comparison.

Run from the repository root after installing the package:
    python examples/06_normal_and_nig.py
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import stochPathSim as sps
from stochPathSim.analytics import bachelier_greeks, bachelier_price, nig_price
from stochPathSim.pricing import EuropeanCall, price_mc

# Normal volatility is in absolute forward units. A forward has zero drift
# under its forward measure, even though the discount rate is nonzero.
forward, strike, T, rate, normal_vol = -0.005, 0.0, 1.0, 0.015, 0.01
normal = sps.ArithmeticBrownianMotion(forward, mu=0, sigma=normal_vol)
normal_mc = price_mc(normal, EuropeanCall(strike), T, discount=rate,
                     n_steps=1, n_paths=100_000, seed=42, antithetic=True)
normal_exact = bachelier_price(forward, strike, T, np.exp(-rate*T), normal_vol)
print(f"Normal call: MC {normal_mc.price:.7f} +/- {normal_mc.stderr:.7f}; formula {normal_exact:.7f}")
print("Forward Greeks:", bachelier_greeks(forward, strike, T, np.exp(-rate*T), normal_vol))

# NIG delta is an activity parameter, not a Gaussian volatility or an option Greek.
params = dict(alpha=8.0, beta=-3.0, delta=0.25)
nig = sps.NormalInverseGaussian(s0=100, mu=rate, q=0.01, **params)
nig_mc = price_mc(nig, EuropeanCall(105), T, discount=rate, n_steps=1,
                  n_paths=100_000, seed=42, antithetic=True)
nig_exact = nig_price(100, 105, T, rate, **params, q=0.01)
print(f"NIG call: MC {nig_mc.price:.5f} +/- {nig_mc.stderr:.5f}; NIG CDF price {nig_exact:.5f}")

# Equal log-return variance isolates the distribution-shape comparison.
sigma_gbm = np.sqrt(nig.log_return_variance(1.0))
gbm = sps.GBM(100, rate, sigma_gbm, q=0.01)
nig_result = nig.simulate(T=T, n_steps=64, n_paths=5_000, seed=13)
gbm_result = gbm.simulate(T=T, n_steps=64, n_paths=5_000, seed=13)
normal_result = normal.simulate(T=T, n_steps=64, n_paths=40, seed=13)
fig = make_subplots(rows=1, cols=2, subplot_titles=("Normal forward paths (negative values allowed)",
                                                  "Log returns: matched variance, different shape"))
for i, path in enumerate(normal_result["spot"][:15]):
    fig.add_trace(go.Scatter(x=normal_result.times, y=path, mode="lines", opacity=0.45,
                             name="Normal forward", legendgroup="normal", showlegend=i == 0,
                             line=dict(color="#0A66C2", width=1)), row=1, col=1)
for label, result, color in [("NIG", nig_result, "#d97706"), ("GBM", gbm_result, "#0f766e")]:
    fig.add_trace(go.Histogram(x=np.log(result.terminal()/100), histnorm="probability density",
                               nbinsx=80, opacity=0.55, name=label, marker_color=color), row=1, col=2)
fig.update_layout(template="plotly_white", barmode="overlay", height=480,
                  title="stochPathSim: normal forwards and NIG returns (simulated data)")
fig.update_xaxes(title_text="Years", row=1, col=1)
fig.update_yaxes(title_text="Forward", row=1, col=1)
fig.update_xaxes(title_text="log(S_T / S_0)", row=1, col=2)
fig.update_yaxes(title_text="Density", row=1, col=2)
output = Path(__file__).parent / "output" / "normal-and-nig.html"
output.parent.mkdir(exist_ok=True)
fig.write_html(output, include_plotlyjs=True)
print(f"Interactive chart: {output}")
