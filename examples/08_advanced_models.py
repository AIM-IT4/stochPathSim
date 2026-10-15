"""Advanced models: GARCH/GJR, fractional Brownian motion, Sobol, portfolio attribution.

Run from repository root:
    python examples/08_advanced_models.py

Creates four self-contained interactive Plotly HTML figures in examples/output/.
All data in the example are synthetic.
"""

from pathlib import Path

import numpy as np

import stochPathSim as sps
from stochPathSim import viz
from stochPathSim.core import sobol_normals
from stochPathSim.risk import normal_portfolio_risk

output = Path(__file__).parent / "output"
output.mkdir(exist_ok=True)

# GJR-GARCH reacts more strongly to a negative innovation than GARCH.
common_shocks = np.random.default_rng(12).normal(size=(1000, 252, 1))
garch = sps.GARCH11(100, 0.03, omega=0.0008, alpha=0.07, beta=0.9, v0=0.04)
gjr = sps.GJRGARCH11(100, 0.03, omega=0.0008, alpha=0.05,
                      beta=0.88, gamma=0.1, v0=0.04)
plain = garch.simulate(n_steps=252, normals=common_shocks)
leverage = gjr.simulate(n_steps=252, normals=common_shocks)
fig = viz.paths(leverage, name="variance", n_paths=15)
fig.update_layout(title="GJR-GARCH: volatility clustering (synthetic paths)")
fig.write_html(output / "gjr-volatility.html", include_plotlyjs=True)
print("Terminal mean variance, GARCH / GJR:", plain["variance"][:, -1].mean(),
      leverage["variance"][:, -1].mean())

# Hurst exponent controls persistence and roughness, not Gaussian variance alone.
fbm = sps.FractionalBrownianMotion(x0=0, hurst=0.25, sigma=1.0)
fpaths = fbm.simulate(T=1, n_steps=80, n_paths=150, seed=7)
fig = viz.paths(fpaths, name="value", n_paths=25)
fig.update_layout(title="Fractional Brownian motion (H = 0.25)")
fig.write_html(output / "fractional-brownian.html", include_plotlyjs=True)

# Scrambled Sobol with a Brownian bridge can feed any Gaussian-factor simulator.
z = sobol_normals(4096, 32, seed=5, brownian_bridge=True)
qmc_result = sps.GBM(100, 0.03, 0.20).simulate(T=1, n_steps=32, normals=z)
fig = viz.fan(qmc_result)
fig.update_layout(title="GBM: Sobol / Brownian-bridge scenario fan")
fig.write_html(output / "sobol-fan.html", include_plotlyjs=True)

# Signed Euler contributions correctly handle hedging positions.
covariance = np.array([[0.04, 0.012, 0.004],
                       [0.012, 0.09, 0.008],
                       [0.004, 0.008, 0.025]])
portfolio = normal_portfolio_risk(weights=[1.0, -0.2, 0.7],
                                  covariance=covariance, level=0.99)
fig = viz.portfolio_risk_chart(portfolio, labels=["Equity", "Rates hedge", "Credit"])
fig.write_html(output / "portfolio-risk-contributions.html", include_plotlyjs=True)
print(f"99% VaR: {portfolio.var:.5f}, ES: {portfolio.es:.5f}")
print("VaR contributions:", portfolio.component_var)
print("ES contributions:", portfolio.component_es)
print("Generated HTML charts:", output)
