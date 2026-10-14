"""Build a full, self-contained HTML validation report for the Heston model.

The report contains:
- Monte Carlo convergence of the QE scheme against the semi-analytic Heston price;
- the discretisation bias of the QE and Euler schemes, with their estimated weak order;
- a comparison of GBM, Heston and Merton terminal distributions;
- a backtest of 10-day returns against realised data generated from a Heston world.

Open examples/output/heston_validation_report.html in a browser; it works offline.
"""

from pathlib import Path

import numpy as np

import stochPathSim as sps
from stochPathSim.analytics import heston_price
from stochPathSim.pricing import EuropeanCall
from stochPathSim.validation import ValidationReport

S0, K, T, r = 100.0, 100.0, 1.0, 0.03
params = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
reference = heston_price(S0, K, T, r, 0.0, **params)
qe = sps.Heston(S0, r, **params)

report = ValidationReport("Heston model validation")

report.add_convergence(
    qe,
    EuropeanCall(K),
    T,
    reference,
    label="Convergence of the QE scheme",
    n_paths_list=(2_000, 8_000, 32_000, 128_000),
    n_steps=25,
    seed=1,
    discount=r,
)

report.add_discretisation(
    qe,
    EuropeanCall(K),
    T,
    reference,
    label="Discretisation: QE scheme",
    n_steps_list=(2, 4, 8, 16, 32),
    n_paths=100_000,
    seed=2,
    discount=r,
)

report.add_discretisation(
    lambda n: sps.Heston(S0, r, **params, scheme="euler"),
    EuropeanCall(K),
    T,
    reference,
    label="Discretisation: full-truncation Euler",
    n_steps_list=(2, 4, 8, 16, 32),
    n_paths=100_000,
    seed=2,
    discount=r,
)

report.add_comparison(
    {
        "GBM (20% vol)": sps.GBM(S0, r, 0.2),
        "Heston": qe,
        "Merton jumps": sps.MertonJumpDiffusion(S0, r, 0.15, 1.0, -0.1, 0.15),
    },
    T=T,
    n_steps=50,
    n_paths=50_000,
    seed=3,
    label="GBM vs Heston vs Merton",
)

# backtest: 10-day spot moves from a Heston "real world" against the Heston model's distribution
horizon, n_dates = 10 / 252, 250
world = qe.simulate(T=horizon, n_steps=10, n_paths=n_dates, seed=99)
realised = world.terminal()
model_dist = qe.simulate(T=horizon, n_steps=10, n_paths=5_000, seed=4).terminal()
report.add_backtest(
    np.tile(model_dist, (n_dates, 1)), realised, level=0.99, label="Backtest of 10-day spot moves"
)

out = report.to_html(Path(__file__).resolve().parent / "output" / "heston_validation_report.html")
print(report.summary().to_string(index=False))
print(f"\nall checks passed: {report.passed}")
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
