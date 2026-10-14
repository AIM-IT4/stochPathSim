"""Rough volatility: the rough Bergomi model."""

from __future__ import annotations

import numpy as np
from scipy.special import hyp2f1

from ..core.process import StochasticProcess, check_correlation, check_positive


def _volterra_cov(u: np.ndarray, v: np.ndarray, hurst: float) -> np.ndarray:
    r"""Covariance of the Volterra process :math:`\tilde Y_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2} dW_s`.

    For ``u <= v``, :math:`2H \int_0^u (u-s)^a (v-s)^a ds` with :math:`a = H - 1/2`, which equals
    :math:`2H\, d^a u^{a+1}/(a+1)\; {}_2F_1(-a, a+1; a+2; -u/d)` with ``d = v - u``.
    """
    a = hurst - 0.5
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    d = hi - lo
    out = np.empty(np.broadcast(lo, hi).shape)
    same = d <= 1e-14 * np.maximum(hi, 1.0)
    out[same] = lo[same] ** (2 * a + 1) / (2 * a + 1)
    ds, ls = d[~same], lo[~same]
    out[~same] = ds**a * ls ** (a + 1) / (a + 1) * hyp2f1(-a, a + 1, a + 2, -ls / ds)
    return 2 * hurst * out


class RoughBergomi(StochasticProcess):
    r"""Rough Bergomi model (Bayer, Friz and Gatheral, 2016)

    .. math::

        \frac{dS_t}{S_t} = \mu\,dt + \sqrt{V_t}\,dZ_t,\qquad
        V_t = \xi_0 \exp\!\Big(\eta \tilde Y_t - \tfrac12 \eta^2 t^{2H}\Big),

        \tilde Y_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\,dW_s,

        dZ_t = \rho\,dW_t + \sqrt{1-\rho^2}\,dW^\perp_t

    with a flat forward variance curve ``xi0`` and Hurst exponent ``0 < H < 1/2``
    (rough volatility).

    The Volterra process and the Brownian motion driving it are sampled
    **exactly** on the grid from their joint Gaussian law (Cholesky factor of the
    ``2n x 2n`` covariance), so ``E[V_t] = xi0`` holds without discretisation bias.
    The log-spot uses the left-point variance on each step, which keeps ``S`` an
    exact martingale (``E[S_T] = s0 exp(mu T)``). Cost grows as ``n_steps^2``, so
    keep ``n_steps`` in the low hundreds.

    Parameters
    ----------
    s0 : float
        Initial spot.
    mu : float
        Drift of the spot (risk-free rate minus dividend yield for pricing).
    xi0 : float
        Flat forward variance.
    eta : float
        Volatility of volatility.
    hurst : float
        Hurst exponent ``H`` in ``(0, 1/2]``; ``0.1`` is typical for equity indices.
    rho : float
        Correlation between spot and volatility shocks.
    """

    state_names = ("spot", "variance")
    n_factors = 3

    def __init__(self, s0: float, mu: float, xi0: float, eta: float, hurst: float, rho: float):
        check_positive(s0=s0, xi0=xi0, eta=eta)
        check_correlation(rho)
        if not 0.0 < hurst <= 0.5:
            raise ValueError(f"hurst must lie in (0, 0.5], got {hurst}")
        self.s0 = float(s0)
        self.mu = float(mu)
        self.xi0 = float(xi0)
        self.eta = float(eta)
        self.hurst = float(hurst)
        self.rho = float(rho)

    def _joint_factor(self, times: np.ndarray) -> np.ndarray:
        """Lower Cholesky factor of the covariance of ``(Y_t1..Y_tn, W_t1..W_tn)``."""
        n = times.size
        a = self.hurst - 0.5
        u, v = np.meshgrid(times, times, indexing="ij")
        cov_yy = _volterra_cov(u, v, self.hurst)
        m = np.minimum(u, v)
        cov_yw = np.sqrt(2 * self.hurst) * (u ** (a + 1) - (u - m) ** (a + 1)) / (a + 1)
        cov = np.empty((2 * n, 2 * n))
        cov[:n, :n] = cov_yy
        cov[:n, n:] = cov_yw
        cov[n:, :n] = cov_yw.T
        cov[n:, n:] = m
        jitter = 1e-14 * np.trace(cov) / (2 * n)
        for _ in range(6):
            try:
                return np.linalg.cholesky(cov + jitter * np.eye(2 * n))
            except np.linalg.LinAlgError:
                jitter *= 100.0
        raise np.linalg.LinAlgError("rough Bergomi covariance is not positive definite")

    def _simulate(self, grid, normals, rng):
        n_paths, n_steps, _ = normals.shape
        times = grid.times[1:]
        lower = self._joint_factor(times)
        z = np.concatenate([normals[:, :, 0], normals[:, :, 1]], axis=1)
        joint = z @ lower.T
        y = joint[:, :n_steps]
        w = joint[:, n_steps:]
        dw = np.diff(np.concatenate([np.zeros((n_paths, 1)), w], axis=1), axis=1)

        variance = np.empty((n_paths, n_steps + 1))
        variance[:, 0] = self.xi0
        variance[:, 1:] = self.xi0 * np.exp(self.eta * y - 0.5 * self.eta**2 * times ** (2 * self.hurst))

        dt = grid.dt
        dz = self.rho * dw + np.sqrt(1.0 - self.rho**2) * np.sqrt(dt) * normals[:, :, 2]
        v_left = variance[:, :-1]
        log_inc = (self.mu - 0.5 * v_left) * dt + np.sqrt(v_left) * dz
        log_s = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
        return {"spot": self.s0 * np.exp(log_s), "variance": variance}

    def mean(self, t):
        """Expected spot ``E[S_t] = s0 exp(mu t)``."""
        return self.s0 * np.exp(self.mu * np.asarray(t, dtype=float))

    def variance_mean(self, t):
        """Expected instantaneous variance, equal to ``xi0`` at every time."""
        return np.full_like(np.asarray(t, dtype=float), self.xi0)
