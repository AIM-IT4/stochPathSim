"""Helpers for the version 1 interface."""

from __future__ import annotations

from .core import TimeGrid


def legacy_grid(T: float, n: int) -> TimeGrid:
    """Version 1 used ``n`` time points from 0 to ``T`` inclusive."""
    if int(n) < 2:
        raise ValueError("n must be at least 2")
    return TimeGrid.uniform(T, int(n) - 1)
