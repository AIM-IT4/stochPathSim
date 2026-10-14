"""Cross-sectional statistics of simulated paths and band backtests.

These are the tools behind risk-factor simulation backtesting: for each
horizon, compare the simulated distribution with what was realised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.result import SimulationResult


def _as_paths(result_or_array, name=None):
    """Return ``(times, paths)`` from a SimulationResult or a 2-d array."""
    if isinstance(result_or_array, SimulationResult):
        return result_or_array.times, result_or_array.get(name)
    paths = np.asarray(result_or_array, dtype=float)
    if paths.ndim != 2:
        raise ValueError("paths must be a 2-d array of shape (n_paths, n_times)")
    return np.arange(paths.shape[1], dtype=float), paths


def _quantile_label(q: float) -> str:
    return f"q{round(q * 100):02d}"


def path_statistics(
    result_or_array,
    reference=None,
    quantiles=(0.01, 0.10, 0.90, 0.99),
    name: str | None = None,
) -> pd.DataFrame:
    """Mean, standard deviation, P-level and quantiles at each time.

    Parameters
    ----------
    result_or_array : SimulationResult or ndarray
        Simulated paths, shape ``(n_paths, n_times)`` for an array.
    reference : float or array_like, optional
        Level against which ``p_level`` is measured: the share of paths above
        it at each time. Defaults to the initial value of the paths.
    quantiles : sequence of float
        Quantile levels in [0, 1]; columns are named ``q01``, ``q99`` etc.
    name : str, optional
        State variable to use when a SimulationResult holds several.

    Returns
    -------
    pandas.DataFrame
        Indexed by time.
    """
    times, paths = _as_paths(result_or_array, name)
    if reference is None:
        reference = paths[:, 0].mean()
    reference = np.broadcast_to(np.asarray(reference, dtype=float), (paths.shape[1],))
    data = {
        "mean": paths.mean(axis=0),
        "std": paths.std(axis=0, ddof=1) if paths.shape[0] > 1 else np.zeros(paths.shape[1]),
        "p_level": (paths > reference).mean(axis=0),
    }
    qs = np.quantile(paths, list(quantiles), axis=0)
    for q, values in zip(quantiles, qs, strict=True):
        data[_quantile_label(q)] = values
    return pd.DataFrame(data, index=pd.Index(times, name="time"))


def exceedance_counts(simulated, realised, lower: float = 0.01, upper: float = 0.99, name=None) -> dict:
    """Count how often realised values fall outside the simulated quantile band.

    For each time step the band ``[q_lower, q_upper]`` of the simulated
    distribution is compared with the realised value(s). This is the usual
    band backtest for risk-factor models. (The legacy ``calculate_statistics``
    compared the *initial* value with the band, which says nothing about model
    quality; this function compares realisations.)

    Parameters
    ----------
    simulated : SimulationResult or ndarray
        Simulated paths ``(n_paths, n_times)``.
    realised : array_like
        One realised path ``(n_times,)`` or several ``(n_realisations, n_times)``.
    lower, upper : float
        Quantile levels of the band.

    Returns
    -------
    dict
        ``upper`` and ``lower`` exceedance counts, ``n_obs``, the expected
        counts under a correct model, and boolean ``upper_mask``/``lower_mask``.
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("need 0 <= lower < upper <= 1")
    _, paths = _as_paths(simulated, name)
    realised = np.atleast_2d(np.asarray(realised, dtype=float))
    if realised.shape[1] != paths.shape[1]:
        raise ValueError("realised paths must have one value per simulated time")
    q_lo, q_hi = np.quantile(paths, [lower, upper], axis=0)
    upper_mask = realised > q_hi
    lower_mask = realised < q_lo
    n_obs = realised.size
    return {
        "upper": int(upper_mask.sum()),
        "lower": int(lower_mask.sum()),
        "n_obs": int(n_obs),
        "expected_upper": n_obs * (1.0 - upper),
        "expected_lower": n_obs * lower,
        "upper_mask": upper_mask,
        "lower_mask": lower_mask,
    }
