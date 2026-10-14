"""Counterparty exposure of an FX forward: EE, PFE, effective EPE and a simple CVA."""

import numpy as np

import stochPathSim as sps
from stochPathSim.risk import ExposureProfile, collateralised_exposure, cva_from_exposure

T, strike, notional, r = 2.0, 1.10, 1_000_000, 0.03
fx = sps.GBM(s0=1.10, mu=0.0, sigma=0.10).simulate(T=T, n_steps=104, n_paths=20_000, seed=11)

# mark-to-market of a long forward, discounted at a flat rate
tau = T - fx.times
mtm = notional * (fx.get() - strike) * np.exp(-r * tau)

profile = ExposureProfile.from_mtm(mtm, fx.times, pfe_level=0.95)
print(profile.to_frame().iloc[::13].round(0))
print(f"Peak PFE(95%): {profile.max_pfe:,.0f}")

collateralised = collateralised_exposure(mtm, threshold=50_000, margin_period_steps=1)
print(f"Peak PFE(95%) with VM, 50k threshold: {np.quantile(collateralised, 0.95, axis=0).max():,.0f}")

cva = cva_from_exposure(profile.ee, fx.times, hazard_rate=0.02, recovery=0.4, discount_rate=r)
print(f"Unilateral CVA: {cva:,.0f}")
