"""Generate the figure shown in the README."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stochPathSim as sps
from stochPathSim.plotting import plot_fan, plot_paths

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
heston = sps.Heston(s0=100, mu=0.03, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
plot_paths(heston.simulate(T=1, n_steps=252, n_paths=5_000, seed=1), n_paths=25, ax=axes[0],
           title="Heston: spot paths")
curve = sps.ZeroCurve([0.25, 1, 2, 5, 10], [0.030, 0.032, 0.035, 0.038, 0.040])
plot_fan(sps.HullWhite(0.05, 0.012, curve).simulate(T=10, n_steps=120, n_paths=20_000, seed=2),
         ax=axes[1], title="Hull-White: short-rate fan")
plot_paths(sps.MertonJumpDiffusion(100, 0.03, 0.15, 1.0, -0.1, 0.15).simulate(
    T=1, n_steps=252, n_paths=5_000, seed=3), n_paths=25, ax=axes[2], title="Merton jump diffusion")
fig.tight_layout()
fig.savefig("../docs/assets/stochpathsim-paths.png", dpi=130)
