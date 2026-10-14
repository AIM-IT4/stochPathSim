"""Hull-White fitted to an upward-sloping curve: Monte Carlo bond prices reproduce the curve."""

import numpy as np

import stochPathSim as sps
from stochPathSim.analytics import hull_white_caplet

curve = sps.ZeroCurve([0.5, 1, 2, 5, 10], [0.030, 0.032, 0.035, 0.038, 0.040])
model = sps.HullWhite(a=0.05, sigma=0.01, curve=curve)
paths = model.simulate(T=10.0, n_steps=120, n_paths=50_000, seed=7)

for t in (1.0, 5.0, 10.0):
    mc = paths.at(t, "discount").mean()
    print(f"P(0,{t:>4}) curve {curve.discount(t):.5f}  Monte Carlo {mc:.5f}")

print(f"1y x 3m caplet at 4%: {hull_white_caplet(1.0, 1.25, 0.04, 0.05, 0.01, curve):.6f}")
print(f"Short-rate 99% quantile at 10y: {np.quantile(paths.terminal('rate'), 0.99):.4%}")
