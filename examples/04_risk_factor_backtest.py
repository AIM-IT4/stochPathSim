"""Backtest a risk-factor model: PIT values, Anderson-Darling and Kupiec tests.

This is the workflow stochPathSim was originally built for: simulate a risk factor
over a horizon, then check realised outcomes against the simulated distribution.
"""

import numpy as np

import stochPathSim as sps
from stochPathSim.risk import anderson_darling_uniform, kupiec_pof, pit_values

rng = np.random.default_rng(0)
horizon, n_dates = 10 / 252, 500

# "realised" 10-day log returns from a 25% vol world
realised = 100 * np.exp(rng.normal(-0.5 * 0.25**2 * horizon, 0.25 * np.sqrt(horizon), n_dates))

for model_vol in (0.25, 0.15):
    sims = sps.GBM(100, 0.0, model_vol).simulate(T=horizon, n_steps=1, n_paths=5_000, seed=1).terminal()
    pit = pit_values(np.tile(sims, (n_dates, 1)), realised)
    ad, p_ad = anderson_darling_uniform(pit)
    exceed = int(np.sum(pit > 0.99) + np.sum(pit < 0.01))
    lr, p_k = kupiec_pof(n_dates, exceed, level=0.98)
    print(f"model vol {model_vol:.0%}: AD p-value {p_ad:.3f}, exceedances {exceed} (Kupiec p {p_k:.3f})")
