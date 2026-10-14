"""Payoffs evaluated on simulated paths.

Every payoff maps simulated paths to one undiscounted payoff per path at
maturity. Call a payoff with a :class:`~stochPathSim.core.SimulationResult`,
or with a raw ``(n_paths, n_times)`` array plus the matching ``times``.
"""

from __future__ import annotations

import numpy as np

from ..core.result import SimulationResult

_OPTION_TYPES = ("call", "put")
_BARRIER_KINDS = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")

# Broadie-Glasserman-Kou continuity correction constant: -zeta(1/2) / sqrt(2 pi)
_BGK_BETA = 0.5826


def _check_option_type(option_type: str) -> str:
    option_type = str(option_type).lower()
    if option_type not in _OPTION_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return option_type


def _vanilla(values: np.ndarray, K: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(values - K, 0.0)
    return np.maximum(K - values, 0.0)


class Payoff:
    """Base class for path-dependent payoffs.

    Parameters
    ----------
    state : str, optional
        Name of the state variable to read; defaults to the primary state.
    """

    def __init__(self, state: str | None = None):
        self.state = state

    def __call__(self, result, times=None) -> np.ndarray:
        """Undiscounted payoff per path.

        Parameters
        ----------
        result : SimulationResult or ndarray
            Simulated paths. A raw array must have shape ``(n_paths, n_times)``.
        times : array_like, optional
            Grid times for a raw array (needed by time-aware payoffs).
        """
        if isinstance(result, SimulationResult):
            paths = result.get(self.state)
            times = result.times
        else:
            paths = np.asarray(result, dtype=float)
            if paths.ndim == 1:
                paths = paths[:, None]
            if paths.ndim != 2:
                raise ValueError("paths must have shape (n_paths, n_times)")
            if times is None:
                times = np.linspace(0.0, 1.0, paths.shape[1])
            times = np.asarray(times, dtype=float)
            if times.size != paths.shape[1]:
                raise ValueError("times must match the number of path columns")
        return self.evaluate(paths, times)

    def evaluate(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Payoff per path from a ``(n_paths, n_times)`` array."""
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if v is not None)
        return f"{type(self).__name__}({params})"


class European(Payoff):
    """European call or put on the terminal value: ``max(S_T - K, 0)`` or ``max(K - S_T, 0)``."""

    def __init__(self, K: float, option_type: str = "call", state: str | None = None):
        super().__init__(state)
        self.K = float(K)
        self.option_type = _check_option_type(option_type)

    def evaluate(self, paths, times):
        return _vanilla(paths[:, -1], self.K, self.option_type)


class EuropeanCall(European):
    """European call ``max(S_T - K, 0)``."""

    def __init__(self, K: float, state: str | None = None):
        super().__init__(K, "call", state)


class EuropeanPut(European):
    """European put ``max(K - S_T, 0)``."""

    def __init__(self, K: float, state: str | None = None):
        super().__init__(K, "put", state)


class Digital(Payoff):
    """Cash-or-nothing digital paying ``cash`` if ``S_T > K`` (call) or ``S_T < K`` (put)."""

    def __init__(self, K: float, option_type: str = "call", cash: float = 1.0, state: str | None = None):
        super().__init__(state)
        self.K = float(K)
        self.option_type = _check_option_type(option_type)
        self.cash = float(cash)

    def evaluate(self, paths, times):
        s = paths[:, -1]
        hit = s > self.K if self.option_type == "call" else s < self.K
        return self.cash * hit.astype(float)


def _fixing_indices(times: np.ndarray, fixing_times, start_index: int) -> np.ndarray:
    if fixing_times is None:
        if not 0 <= start_index < times.size:
            raise ValueError("start_index is outside the time grid")
        return np.arange(start_index, times.size)
    fixing_times = np.atleast_1d(np.asarray(fixing_times, dtype=float))
    idx = np.searchsorted(times, fixing_times)
    idx = np.clip(idx, 0, times.size - 1)
    lower = np.clip(idx - 1, 0, times.size - 1)
    idx = np.where(np.abs(times[lower] - fixing_times) < np.abs(times[idx] - fixing_times), lower, idx)
    if np.any(np.abs(times[idx] - fixing_times) > 1e-10):
        raise ValueError("every fixing time must be a grid time; build the grid with TimeGrid.with_times")
    return idx


class _Asian(Payoff):
    def __init__(self, K, option_type="call", fixing_times=None, start_index=1, state=None):
        super().__init__(state)
        self.K = float(K)
        self.option_type = _check_option_type(option_type)
        self.fixing_times = fixing_times
        self.start_index = int(start_index)

    def _fixings(self, paths, times):
        return paths[:, _fixing_indices(times, self.fixing_times, self.start_index)]


class AsianArithmetic(_Asian):
    """Fixed-strike Asian option on the arithmetic average of the fixings.

    Parameters
    ----------
    K : float
        Strike.
    option_type : {"call", "put"}
    fixing_times : array_like, optional
        Averaging times (must be grid times). By default every grid time from
        ``start_index`` on (``start_index=1`` excludes ``t = 0``).
    """

    def evaluate(self, paths, times):
        return _vanilla(self._fixings(paths, times).mean(axis=1), self.K, self.option_type)


class AsianGeometric(_Asian):
    """Fixed-strike Asian option on the geometric average of the fixings (see :class:`AsianArithmetic`)."""

    def evaluate(self, paths, times):
        fixings = self._fixings(paths, times)
        if np.any(fixings <= 0.0):
            raise ValueError("geometric averaging needs positive fixings")
        return _vanilla(np.exp(np.log(fixings).mean(axis=1)), self.K, self.option_type)


class Barrier(Payoff):
    """Single-barrier knock-in or knock-out option, monitored at every grid time.

    Parameters
    ----------
    K : float
        Strike.
    barrier : float
        Barrier level.
    kind : {"up-and-out", "up-and-in", "down-and-out", "down-and-in"}
    option_type : {"call", "put"}
    rebate : float
        Paid at maturity if the option is knocked out, or never knocked in.
    sigma : float, optional
        Volatility for the Broadie-Glasserman-Kou continuity correction. When
        given, the barrier is moved towards the spot by the factor
        ``exp(0.5826 sigma sqrt(dt))``, so that discrete monitoring on the grid
        approximates a continuously monitored barrier at ``barrier``.
    """

    def __init__(self, K, barrier, kind, option_type="call", rebate=0.0, sigma=None, state=None):
        super().__init__(state)
        kind = str(kind).lower()
        if kind not in _BARRIER_KINDS:
            raise ValueError(f"kind must be one of {_BARRIER_KINDS}, got {kind!r}")
        if not barrier > 0:
            raise ValueError("barrier must be positive")
        if sigma is not None and not sigma >= 0:
            raise ValueError("sigma must be non-negative")
        self.K = float(K)
        self.barrier = float(barrier)
        self.kind = kind
        self.option_type = _check_option_type(option_type)
        self.rebate = float(rebate)
        self.sigma = sigma

    def effective_barrier(self, times) -> float:
        """Barrier used on the grid, after the optional continuity correction."""
        if self.sigma is None:
            return self.barrier
        dt = float(np.mean(np.diff(times)))
        shift = np.exp(_BGK_BETA * self.sigma * np.sqrt(dt))
        return self.barrier / shift if self.kind.startswith("up") else self.barrier * shift

    def knocked(self, paths, times) -> np.ndarray:
        """Boolean per path: whether the barrier was touched on the grid."""
        h = self.effective_barrier(times)
        if self.kind.startswith("up"):
            return paths.max(axis=1) >= h
        return paths.min(axis=1) <= h

    def evaluate(self, paths, times):
        vanilla = _vanilla(paths[:, -1], self.K, self.option_type)
        hit = self.knocked(paths, times)
        alive = ~hit if self.kind.endswith("out") else hit
        return np.where(alive, vanilla, self.rebate)


class LookbackFixed(Payoff):
    """Fixed-strike lookback: ``max(max S - K, 0)`` (call) or ``max(K - min S, 0)`` (put), over the grid."""

    def __init__(self, K: float, option_type: str = "call", state: str | None = None):
        super().__init__(state)
        self.K = float(K)
        self.option_type = _check_option_type(option_type)

    def evaluate(self, paths, times):
        if self.option_type == "call":
            return np.maximum(paths.max(axis=1) - self.K, 0.0)
        return np.maximum(self.K - paths.min(axis=1), 0.0)


class LookbackFloating(Payoff):
    """Floating-strike lookback: ``S_T - min S`` (call) or ``max S - S_T`` (put), over the grid."""

    def __init__(self, option_type: str = "call", state: str | None = None):
        super().__init__(state)
        self.option_type = _check_option_type(option_type)

    def evaluate(self, paths, times):
        if self.option_type == "call":
            return paths[:, -1] - paths.min(axis=1)
        return paths.max(axis=1) - paths[:, -1]


class Basket(Payoff):
    """Option on a weighted sum of terminal values of several state variables.

    Parameters
    ----------
    weights : array_like
        One weight per state.
    K : float
        Strike.
    option_type : {"call", "put"}
    states : sequence of str, optional
        State names, in the order of ``weights``. Defaults to every state of the
        result, in order. Requires a :class:`SimulationResult`.
    """

    def __init__(self, weights, K: float, option_type: str = "call", states=None):
        super().__init__(None)
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.ndim != 1 or self.weights.size == 0:
            raise ValueError("weights must be a non-empty 1-d array")
        self.K = float(K)
        self.option_type = _check_option_type(option_type)
        self.states = None if states is None else tuple(states)
        if self.states is not None and len(self.states) != self.weights.size:
            raise ValueError("states and weights must have the same length")

    def __call__(self, result, times=None) -> np.ndarray:
        if not isinstance(result, SimulationResult):
            raise TypeError("basket payoffs need a SimulationResult with named states")
        states = self.states if self.states is not None else result.names
        if len(states) != self.weights.size:
            raise ValueError(f"{self.weights.size} weights given for {len(states)} states")
        terminal = np.column_stack([result.terminal(s) for s in states])
        return _vanilla(terminal @ self.weights, self.K, self.option_type)


class Spread(Basket):
    """Spread option on ``S_a(T) - S_b(T)``: ``max(S_a - S_b - K, 0)`` for a call."""

    def __init__(self, K: float = 0.0, option_type: str = "call", states=None):
        if states is not None and len(states) != 2:
            raise ValueError("a spread needs exactly two states")
        super().__init__([1.0, -1.0], K, option_type, states)
