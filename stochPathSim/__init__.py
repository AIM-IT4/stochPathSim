"""stochPathSim: stochastic path simulation, pricing and risk analytics for quantitative finance.

>>> import stochPathSim as sps
>>> paths = sps.Heston(s0=100, mu=0.03, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7).simulate(
...     T=1.0, n_steps=252, n_paths=10_000, seed=42)
"""

from . import analytics, calibration, core, pricing, processes, risk
from ._version import __version__
from .core import SimulationResult, StochasticProcess, TimeGrid
from .curves import ZeroCurve
from .processes import (
    CEV,
    CIR,
    GBM,
    SABR,
    Bates,
    BlackDermanToy,
    BlackKarasinski,
    G2pp,
    Heston,
    HoLee,
    HullWhite,
    MertonJumpDiffusion,
    MultiAssetGBM,
    OrnsteinUhlenbeck,
    VarianceGamma,
    Vasicek,
)

__all__ = [
    "__version__",
    "analytics",
    "calibration",
    "core",
    "pricing",
    "processes",
    "risk",
    "TimeGrid",
    "SimulationResult",
    "StochasticProcess",
    "ZeroCurve",
    "GBM",
    "MultiAssetGBM",
    "MertonJumpDiffusion",
    "Heston",
    "Bates",
    "SABR",
    "CEV",
    "VarianceGamma",
    "OrnsteinUhlenbeck",
    "Vasicek",
    "CIR",
    "HullWhite",
    "HoLee",
    "G2pp",
    "BlackKarasinski",
    "BlackDermanToy",
]
