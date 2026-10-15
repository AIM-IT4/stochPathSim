"""CIR++: a shifted square-root short-rate model fitted to a zero curve."""

from __future__ import annotations

import numpy as np

from ..analytics.rates import cir_zcb
from ..core.process import StochasticProcess
from .rates import CIR, _as_curve


class CIRPlusPlus(StochasticProcess):
    r"""Brigo-Mercurio CIR++ with exact underlying CIR transitions.

    r_t = x_t + phi(t), where x is a standard CIR factor starting at x0,
    and phi(t) = f_market(0,t) - f_CIR(0,t) is a deterministic curve shift.
    This construction fits the initial market discount curve *analytically*.

    States are rate, factor, discount. The factor is non-negative under the
    exact scheme; shifted short rates can be negative even if it is positive.
    Discount factors use a trapezoidal integral of x (as in CIR), plus the
    *exact* deterministic shift integral P_market(0,t)/P_CIR(0,t).
    Thus MC means of discounts approach the input curve with time-step
    refinement; individual discount paths are not continuous-time exact.

    The exact scheme draws noncentral chi-square factors from the seeded
    generator; supplied normals and antithetic shocks do not affect them.
    Use scheme="euler" to drive a full-truncation factor with normals.

    Parameters
    ----------
    x0, kappa, theta, sigma : float
        Parameters of the underlying CIR factor; all must be positive.
    curve : ZeroCurve or float
        Target initial curve, or flat continuously compounded rate.
    scheme : {"exact", "euler"}
        CIR factor transition scheme.
    """

    state_names = ("rate", "factor", "discount")
    n_factors = 1

    def __init__(self, x0: float, kappa: float, theta: float, sigma: float,
                 curve, scheme: str = "exact"):
        if not np.all(np.isfinite([x0, kappa, theta, sigma])):
            raise ValueError("CIR++ parameters must be finite")
        self._cir = CIR(x0, kappa, theta, sigma, scheme=scheme)
        self.x0 = float(x0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.curve = _as_curve(curve)
        self.scheme = scheme

    @property
    def feller_satisfied(self):
        """Whether the underlying CIR factor stays strictly above zero."""
        return self._cir.feller_satisfied

    def _B(self, t):
        """Stable CIR bond loading B(0,t), including B(0)=0."""
        t = np.asarray(t, dtype=float)
        h = np.sqrt(self.kappa**2 + 2 * self.sigma**2)
        e = np.exp(-h * t)
        return 2 * (1 - e) / ((self.kappa + h) + (h - self.kappa) * e)

    def cir_forward_rate(self, t):
        """Initial instantaneous forward f_CIR(0,t) from the CIR affine bond formula."""
        b = self._B(t)
        return (self.kappa * self.theta * b
                + self.x0 * (1 - self.kappa * b - 0.5 * self.sigma**2 * b**2))

    def shift(self, t):
        """Deterministic CIR++ shift phi(t) for the input zero curve."""
        return self.curve.forward_rate(t) - self.cir_forward_rate(t)

    @property
    def r0(self):
        """Initial short rate equals the market instantaneous forward at zero."""
        return float(self.curve.forward_rate(0.0))

    def _simulate(self, grid, normals, rng):
        cir_paths = self._cir._simulate(grid, normals, rng)
        t = grid.times
        factor = cir_paths["rate"]
        rate = factor + self.shift(t)[None, :]
        base_bond = cir_zcb(self.x0, 0.0, t, self.kappa, self.theta, self.sigma)
        discount = cir_paths["discount"] * (self.curve.discount(t) / base_bond)[None, :]
        return {"rate": rate, "factor": factor, "discount": discount}

    def mean(self, t):
        """Exact expected short rate."""
        return self._cir.mean(t) + self.shift(t)

    def variance(self, t):
        """Exact short-rate variance, equal to the CIR factor variance."""
        return self._cir.variance(t)
