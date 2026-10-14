"""Counterparty exposure profiles from simulated mark-to-market values.

All functions take ``mtm`` of shape ``(n_paths, n_times)``: the portfolio value
to us on each path and date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _mtm(mtm) -> np.ndarray:
    v = np.asarray(mtm, dtype=float)
    if v.ndim != 2:
        raise ValueError("mtm must have shape (n_paths, n_times)")
    return v


def expected_exposure(mtm) -> np.ndarray:
    """EE(t) = E[max(V(t), 0)]."""
    return np.maximum(_mtm(mtm), 0.0).mean(axis=0)


def expected_negative_exposure(mtm) -> np.ndarray:
    """ENE(t) = E[min(V(t), 0)] (non-positive)."""
    return np.minimum(_mtm(mtm), 0.0).mean(axis=0)


def potential_future_exposure(mtm, level: float = 0.95) -> np.ndarray:
    """PFE(t): the ``level`` quantile of max(V(t), 0)."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    return np.quantile(np.maximum(_mtm(mtm), 0.0), level, axis=0)


def _time_average(profile, times, horizon):
    times = np.asarray(times, dtype=float)
    profile = np.asarray(profile, dtype=float)
    if times.shape != profile.shape:
        raise ValueError("times and profile must have the same length")
    if horizon is not None:
        mask = times <= horizon + 1e-12
        times, profile = times[mask], profile[mask]
    span = times[-1] - times[0]
    if span <= 0.0:
        return float(profile[0])
    # left-point weights: the value at t_i holds until t_{i+1}
    return float(np.sum(profile[:-1] * np.diff(times)) / span)


def expected_positive_exposure(ee, times, horizon: float | None = None) -> float:
    """EPE: time average of EE over ``[0, horizon]`` (whole grid by default)."""
    return _time_average(ee, times, horizon)


def effective_ee(ee) -> np.ndarray:
    """Effective EE: the running maximum of EE (Basel non-decreasing EE)."""
    return np.maximum.accumulate(np.asarray(ee, dtype=float))


def effective_epe(ee, times, horizon: float = 1.0) -> float:
    """Effective EPE: time average of effective EE over the first ``horizon`` years."""
    return _time_average(effective_ee(ee), times, horizon)


def collateralised_exposure(mtm, threshold: float = 0.0, mta: float = 0.0, margin_period_steps: int = 0):
    """Mark-to-market net of variation margin with a margin period of risk.

    Collateral posted at time ``t`` is based on the value ``margin_period_steps``
    steps earlier: ``C(t) = max(V(t - MPoR) - threshold, 0)`` (and symmetrically
    for negative values), set to zero when below the minimum transfer amount.
    No collateral is held before the first margin call.

    Returns the collateralised value ``V(t) - C(t)``.
    """
    v = _mtm(mtm)
    if margin_period_steps < 0:
        raise ValueError("margin_period_steps must be non-negative")
    if threshold < 0.0 or mta < 0.0:
        raise ValueError("threshold and mta must be non-negative")
    lagged = np.zeros_like(v)
    k = int(margin_period_steps)
    lagged[:, k:] = v[:, : v.shape[1] - k] if k else v
    if k:
        lagged[:, :k] = 0.0
    collateral = np.sign(lagged) * np.maximum(np.abs(lagged) - threshold, 0.0)
    collateral[np.abs(collateral) < mta] = 0.0
    return v - collateral


@dataclass
class ExposureProfile:
    """Exposure metrics of a simulated portfolio."""

    times: np.ndarray
    ee: np.ndarray
    ene: np.ndarray
    pfe: np.ndarray
    effective_ee: np.ndarray
    epe: float
    effective_epe: float
    pfe_level: float = 0.95

    @classmethod
    def from_mtm(cls, mtm, times, pfe_level: float = 0.95, horizon: float = 1.0) -> ExposureProfile:
        """Compute all metrics from simulated values."""
        times = np.asarray(times, dtype=float)
        ee = expected_exposure(mtm)
        return cls(
            times=times,
            ee=ee,
            ene=expected_negative_exposure(mtm),
            pfe=potential_future_exposure(mtm, pfe_level),
            effective_ee=effective_ee(ee),
            epe=expected_positive_exposure(ee, times),
            effective_epe=effective_epe(ee, times, horizon),
            pfe_level=pfe_level,
        )

    @property
    def max_pfe(self) -> float:
        return float(self.pfe.max())

    def to_frame(self):
        """Profiles as a ``pandas.DataFrame`` indexed by time."""
        import pandas as pd

        return pd.DataFrame(
            {
                "EE": self.ee,
                "ENE": self.ene,
                f"PFE{round(self.pfe_level * 100)}": self.pfe,
                "EffectiveEE": self.effective_ee,
            },
            index=pd.Index(self.times, name="time"),
        )


def cva_from_exposure(
    ee,
    times,
    hazard_rate=None,
    survival=None,
    recovery: float = 0.4,
    discount_rate: float = 0.0,
    discount_curve=None,
) -> float:
    """Unilateral CVA ``(1 - R) * sum_i D(t_i) EE(t_i) [S(t_{i-1}) - S(t_i)]``.

    Parameters
    ----------
    ee : array_like
        Expected exposure on ``times``.
    hazard_rate : float, optional
        Flat hazard rate; survival is ``exp(-h t)``.
    survival : callable or array_like, optional
        Survival probabilities on ``times`` (or a function of time) instead of a hazard rate.
    recovery : float
        Recovery rate.
    discount_rate, discount_curve
        Flat continuously compounded rate, or an object with ``discount(t)``
        (e.g. :class:`stochPathSim.curves.ZeroCurve`).
    """
    times = np.asarray(times, dtype=float)
    ee = np.asarray(ee, dtype=float)
    if ee.shape != times.shape:
        raise ValueError("ee and times must have the same length")
    if not 0.0 <= recovery <= 1.0:
        raise ValueError("recovery must lie in [0, 1]")
    if survival is None:
        if hazard_rate is None:
            raise ValueError("give either hazard_rate or survival")
        surv = np.exp(-float(hazard_rate) * times)
    elif callable(survival):
        surv = np.asarray(survival(times), dtype=float)
    else:
        surv = np.asarray(survival, dtype=float)
    df = discount_curve.discount(times) if discount_curve is not None else np.exp(-discount_rate * times)
    default_prob = -np.diff(surv)
    return float((1.0 - recovery) * np.sum(df[1:] * ee[1:] * default_prob))
