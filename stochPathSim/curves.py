"""Yield curves used to drive and calibrate interest-rate models."""

from __future__ import annotations

import numpy as np


class ZeroCurve:
    """A continuously compounded zero curve.

    Zero rates are interpolated between the pillars and held flat outside them,
    and discount factors are ``exp(-z(t) t)``.

    Parameters
    ----------
    times : array_like
        Pillar times in years, strictly increasing and positive.
    zero_rates : array_like
        Continuously compounded zero rates at the pillars.
    interpolation : {"monotone_cubic", "linear"}
        ``"monotone_cubic"`` (PCHIP, the default) keeps instantaneous forward rates
        continuous between the pillars, which matters for curve-fitted short-rate models
        such as Hull-White, whose drift follows the forward curve. ``"linear"`` interpolates
        zero rates linearly; its forward curve jumps at every pillar.
    """

    def __init__(self, times, zero_rates, interpolation: str = "monotone_cubic"):
        t = np.asarray(times, dtype=float)
        z = np.asarray(zero_rates, dtype=float)
        if t.ndim != 1 or t.shape != z.shape or t.size == 0:
            raise ValueError("times and zero_rates must be 1-d arrays of equal length")
        if np.any(t <= 0.0) or np.any(np.diff(t) <= 0.0):
            raise ValueError("pillar times must be positive and strictly increasing")
        if interpolation not in ("monotone_cubic", "linear"):
            raise ValueError("interpolation must be 'monotone_cubic' or 'linear'")
        self.times = t
        self.zero_rates = z
        self.interpolation = interpolation
        self._pchip = None
        if interpolation == "monotone_cubic" and t.size > 2:
            from scipy.interpolate import PchipInterpolator

            self._pchip = PchipInterpolator(t, z, extrapolate=False)

    @classmethod
    def flat(cls, rate: float) -> ZeroCurve:
        """A flat curve at a continuously compounded ``rate``."""
        return cls([1.0], [rate])

    @classmethod
    def from_discount_factors(cls, times, discount_factors, **kwargs) -> ZeroCurve:
        """Build a curve from discount factors ``P(0, t)``."""
        t = np.asarray(times, dtype=float)
        df = np.asarray(discount_factors, dtype=float)
        if np.any(df <= 0.0):
            raise ValueError("discount factors must be positive")
        return cls(t, -np.log(df) / t, **kwargs)

    def zero_rate(self, t):
        """Continuously compounded zero rate to time ``t``."""
        t = np.asarray(t, dtype=float)
        if self._pchip is None:
            return np.interp(t, self.times, self.zero_rates)
        inside = np.clip(t, self.times[0], self.times[-1])
        return self._pchip(inside)

    def discount(self, t):
        """Discount factor ``P(0, t)``."""
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t, dt: float = 1e-5):
        """Instantaneous forward rate ``f(0, t) = -d ln P(0, t) / dt``."""
        t = np.asarray(t, dtype=float)
        lo = np.maximum(t - dt, 0.0)
        hi = t + dt
        return -(np.log(self.discount(hi)) - np.log(self.discount(lo))) / (hi - lo)

    def forward_rate_between(self, t1, t2):
        """Simply compounded forward rate between ``t1`` and ``t2``."""
        t1 = np.asarray(t1, dtype=float)
        t2 = np.asarray(t2, dtype=float)
        return (self.discount(t1) / self.discount(t2) - 1.0) / (t2 - t1)

    def __repr__(self) -> str:
        if self.times.size == 1:
            return f"ZeroCurve.flat({self.zero_rates[0]:g})"
        return f"ZeroCurve(pillars={self.times.size}, T={self.times[-1]:g})"
