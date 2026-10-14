"""Generate the figure shown in the README (Plotly; static export needs ``pip install kaleido``)."""

from pathlib import Path

from plotly.subplots import make_subplots

import stochPathSim as sps
from stochPathSim import viz

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"

heston = sps.Heston(s0=100, mu=0.03, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
curve = sps.ZeroCurve([0.25, 1, 2, 5, 10], [0.030, 0.032, 0.035, 0.038, 0.040])
panels = [
    viz.paths(heston.simulate(T=1, n_steps=252, n_paths=5_000, seed=1), n_paths=25),
    viz.short_rate_fan(sps.HullWhite(0.05, 0.012, curve).simulate(T=10, n_steps=120, n_paths=20_000, seed=2)),
    viz.paths(sps.MertonJumpDiffusion(100, 0.03, 0.15, 1.0, -0.1, 0.15).simulate(
        T=1, n_steps=252, n_paths=5_000, seed=3), n_paths=25),
]
titles = ("Heston: spot paths", "Hull-White: short-rate fan", "Merton jump diffusion")

fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.06)
for col, panel in enumerate(panels, start=1):
    for trace in panel.data:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=col)
    fig.update_xaxes(title_text="time (years)", row=1, col=col)
fig.update_yaxes(title_text="spot", row=1, col=1)
fig.update_yaxes(title_text="rate", tickformat=".0%", row=1, col=2)
fig.update_layout(template=viz.template(), height=430, width=1500, margin=dict(l=60, r=20, t=50, b=50),
                  hovermode="x")

OUT.mkdir(parents=True, exist_ok=True)
fig.write_html(OUT / "stochpathsim-paths.html", include_plotlyjs="cdn")
try:
    fig.write_image(OUT / "stochpathsim-paths.png", scale=2)
    print("wrote", OUT / "stochpathsim-paths.png")
except Exception as exc:  # kaleido missing or no Chrome available
    print(f"static export failed ({exc!s:.80}); wrote the interactive HTML only")

# a second image for the validation lab
qe = sps.Heston(100, 0.03, 0.04, 1.5, 0.04, 0.6, -0.7)
r = qe.simulate(T=1, n_steps=50, n_paths=20_000, seed=4)
dens = viz.density_evolution(r, kind="heatmap", n_times=50, title="Heston spot: density over time")
dens.update_layout(height=420, width=900)
try:
    dens.write_image(OUT / "density-evolution.png", scale=2)
except Exception:
    pass
