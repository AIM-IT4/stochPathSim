"""Equity and forward models: jump diffusions, stochastic volatility, SABR, CEV and multi-asset GBM."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..core.process import StochasticProcess, check_correlation, check_non_negative, check_positive
from ..core.random import cholesky


def _jump_compensator(jump_mean: float, jump_std: float) -> float:
    """``E[e^J] - 1`` for normally distributed log-jumps ``J``."""
    return np.exp(jump_mean + 0.5 * jump_std**2) - 1.0


def _log_jumps(rng, lam, jump_mean, jump_std, dt, n_paths):
    """Sum of log-jumps over each step, shape ``(n_paths, n_steps)``."""
    counts = rng.poisson(lam * dt, size=(n_paths, dt.size))
    return counts * jump_mean + jump_std * np.sqrt(counts) * rng.standard_normal(counts.shape)


class MertonJumpDiffusion(StochasticProcess):
    r"""Merton (1976) jump diffusion

    .. math:: \frac{dS_t}{S_{t^-}} = (\mu - q - \lambda k)\,dt + \sigma\,dW_t + (e^{J} - 1)\,dN_t

    with Poisson jumps of intensity ``lam`` and normal log-jumps
    ``J ~ N(jump_mean, jump_std^2)``; ``k = E[e^J] - 1`` compensates the jumps so
    that ``E[S_t] = s0 exp((mu - q) t)``. Simulated exactly.
    """

    state_names = ("spot",)
    n_factors = 1

    def __init__(self, s0: float, mu: float, sigma: float, lam: float, jump_mean: float,
                 jump_std: float, q: float = 0.0):
        check_positive(s0=s0)
        check_non_negative(sigma=sigma, lam=lam, jump_std=jump_std)
        self.s0 = float(s0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.jump_mean = float(jump_mean)
        self.jump_std = float(jump_std)
        self.q = float(q)

    def _simulate(self, grid, normals, rng):
        dt = grid.dt
        n_paths = normals.shape[0]
        k = _jump_compensator(self.jump_mean, self.jump_std)
        drift = (self.mu - self.q - self.lam * k - 0.5 * self.sigma**2) * dt
        inc = drift + self.sigma * np.sqrt(dt) * normals[:, :, 0]
        inc += _log_jumps(rng, self.lam, self.jump_mean, self.jump_std, dt, n_paths)
        log_s = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)
        return {"spot": self.s0 * np.exp(log_s)}

    def mean(self, t):
        """Expected value ``E[S_t]``."""
        return self.s0 * np.exp((self.mu - self.q) * np.asarray(t, dtype=float))


class Heston(StochasticProcess):
    r"""Heston (1993) stochastic volatility model

    .. math::
        dS_t &= (\mu - q) S_t\,dt + \sqrt{v_t} S_t\,dW^S_t \\
        dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^v_t, \qquad d\langle W^S, W^v\rangle = \rho\,dt

    Parameters
    ----------
    s0, mu, q : float
        Initial spot, drift and dividend yield.
    v0, kappa, theta, xi, rho : float
        Initial variance, mean-reversion speed, long-run variance, vol of vol and correlation.
    scheme : {"qe", "euler"}
        ``"qe"`` is Andersen's (2008) quadratic-exponential scheme with a martingale-corrected
        log-spot step; ``"euler"`` is a full-truncation Euler scheme.
    """

    state_names = ("spot", "variance")
    n_factors = 2
    _psi_c = 1.5

    def __init__(self, s0: float, mu: float, v0: float, kappa: float, theta: float, xi: float,
                 rho: float, q: float = 0.0, scheme: str = "qe"):
        check_positive(s0=s0, kappa=kappa)
        check_non_negative(v0=v0, theta=theta, xi=xi)
        check_correlation(rho)
        if scheme not in ("qe", "euler"):
            raise ValueError("scheme must be 'qe' or 'euler'")
        self.s0 = float(s0)
        self.mu = float(mu)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.q = float(q)
        self.scheme = scheme

    def _log_jump_increments(self, grid, n_paths, rng):
        return 0.0, None

    def _simulate(self, grid, normals, rng):
        n_paths = normals.shape[0]
        drift_adj, jumps = self._log_jump_increments(grid, n_paths, rng)
        x = np.empty((n_paths, grid.n_steps + 1))
        v = np.empty((n_paths, grid.n_steps + 1))
        x[:, 0] = np.log(self.s0)
        v[:, 0] = self.v0
        step = self._qe_step if self.scheme == "qe" else self._euler_step
        for i, dt in enumerate(grid.dt):
            dx, v[:, i + 1] = step(v[:, i], dt, normals[:, i, 0], normals[:, i, 1])
            x[:, i + 1] = x[:, i] + (self.mu - self.q - drift_adj) * dt + dx
            if jumps is not None:
                x[:, i + 1] += jumps[:, i]
        spot = np.exp(x)
        spot[:, 0] = self.s0
        # the Euler scheme lets the raw variance go negative; report its positive part
        return {"spot": spot, "variance": np.maximum(v, 0.0)}

    def _euler_step(self, v, dt, z_s, z_v):
        vp = np.maximum(v, 0.0)
        sq = np.sqrt(vp * dt)
        w_s = self.rho * z_v + np.sqrt(1.0 - self.rho**2) * z_s
        v_next = v + self.kappa * (self.theta - vp) * dt + self.xi * sq * z_v
        return -0.5 * vp * dt + sq * w_s, v_next

    def _qe_step(self, v, dt, z_s, z_v):
        k, th, xi, rho = self.kappa, self.theta, self.xi, self.rho
        e = np.exp(-k * dt)
        m = th + (v - th) * e
        s2 = v * xi**2 * e / k * (1.0 - e) + th * xi**2 / (2.0 * k) * (1.0 - e) ** 2
        psi = s2 / np.maximum(m * m, 1e-300)
        quad = psi <= self._psi_c
        v_next = np.empty_like(v)

        # quadratic branch
        pq = np.where(quad, psi, 1.0)
        inv = 2.0 / pq
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
        a = m / (1.0 + b2)
        # exponential branch
        pe = np.where(quad, 3.0, psi)
        p = (pe - 1.0) / (pe + 1.0)
        beta = (1.0 - p) / np.maximum(m, 1e-300)
        u = norm.cdf(z_v)
        v_exp = np.where(u <= p, 0.0, np.log((1.0 - p) / np.maximum(1.0 - u, 1e-300)) / beta)
        v_next = np.where(quad, a * (np.sqrt(b2) + z_v) ** 2, v_exp)

        # log-spot step with gamma1 = gamma2 = 1/2 and a martingale correction
        g1 = g2 = 0.5
        if xi > 0.0:
            k1 = g1 * dt * (k * rho / xi - 0.5) - rho / xi
            k2 = g2 * dt * (k * rho / xi - 0.5) + rho / xi
            k0_base = -rho * k * th / xi * dt
        else:
            k1 = k2 = -0.25 * dt
            k0_base = 0.0
        k3 = g1 * dt * (1.0 - rho**2)
        k4 = g2 * dt * (1.0 - rho**2)
        A = k2 + 0.5 * k4
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_m_quad = A * b2 * a / (1.0 - 2.0 * A * a) - 0.5 * np.log(1.0 - 2.0 * A * a)
            log_m_exp = np.log(p + beta * (1.0 - p) / (beta - A))
        ok = np.where(quad, 2.0 * A * a < 1.0, A < beta)
        log_m = np.where(quad, log_m_quad, log_m_exp)
        k0 = np.where(ok, -log_m - (k1 + 0.5 * k3) * v, k0_base)
        dx = k0 + k1 * v + k2 * v_next + np.sqrt(np.maximum(k3 * v + k4 * v_next, 0.0)) * z_s
        return dx, v_next

    def mean(self, t):
        """Expected spot ``E[S_t]``."""
        return self.s0 * np.exp((self.mu - self.q) * np.asarray(t, dtype=float))

    def variance_mean(self, t):
        """Expected variance ``E[v_t]``."""
        t = np.asarray(t, dtype=float)
        return self.theta + (self.v0 - self.theta) * np.exp(-self.kappa * t)


class Bates(Heston):
    """Bates (1996) model: Heston stochastic volatility plus compensated Merton log-normal jumps.

    Parameters are those of :class:`Heston` plus ``lam``, ``jump_mean`` and ``jump_std``
    (see :class:`MertonJumpDiffusion`).
    """

    def __init__(self, s0: float, mu: float, v0: float, kappa: float, theta: float, xi: float,
                 rho: float, lam: float, jump_mean: float, jump_std: float, q: float = 0.0,
                 scheme: str = "qe"):
        super().__init__(s0, mu, v0, kappa, theta, xi, rho, q=q, scheme=scheme)
        check_non_negative(lam=lam, jump_std=jump_std)
        self.lam = float(lam)
        self.jump_mean = float(jump_mean)
        self.jump_std = float(jump_std)

    def _log_jump_increments(self, grid, n_paths, rng):
        k = _jump_compensator(self.jump_mean, self.jump_std)
        return self.lam * k, _log_jumps(rng, self.lam, self.jump_mean, self.jump_std, grid.dt, n_paths)


class SABR(StochasticProcess):
    r"""SABR model (Hagan et al., 2002) for a forward

    .. math::
        dF_t &= \alpha_t F_t^\beta\,dW^F_t \\
        d\alpha_t &= \nu\,\alpha_t\,dW^\alpha_t, \qquad d\langle W^F, W^\alpha\rangle = \rho\,dt

    The volatility is simulated exactly (lognormal); the forward uses an Euler step
    with absorption at zero.
    """

    state_names = ("forward", "vol")
    n_factors = 2

    def __init__(self, f0: float, alpha: float, beta: float, rho: float, nu: float):
        check_positive(f0=f0, alpha=alpha)
        check_non_negative(nu=nu)
        check_correlation(rho)
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        self.f0 = float(f0)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.nu = float(nu)

    def _simulate(self, grid, normals, rng):
        n_paths = normals.shape[0]
        f = np.empty((n_paths, grid.n_steps + 1))
        a = np.empty((n_paths, grid.n_steps + 1))
        f[:, 0] = self.f0
        a[:, 0] = self.alpha
        c = np.sqrt(1.0 - self.rho**2)
        for i, dt in enumerate(grid.dt):
            z_a = normals[:, i, 1]
            z_f = self.rho * z_a + c * normals[:, i, 0]
            fi = f[:, i]
            alive = fi > 0.0
            step = a[:, i] * np.where(alive, np.abs(fi) ** self.beta, 0.0) * np.sqrt(dt) * z_f
            f[:, i + 1] = np.where(alive, np.maximum(fi + step, 0.0), 0.0)
            a[:, i + 1] = a[:, i] * np.exp(self.nu * np.sqrt(dt) * z_a - 0.5 * self.nu**2 * dt)
        return {"forward": f, "vol": a}


class CEV(StochasticProcess):
    r"""Constant elasticity of variance model

    .. math:: dS_t = \mu S_t\,dt + \sigma S_t^{\beta}\,dW_t

    Simulated with an Euler scheme; paths that reach zero are absorbed there.
    ``beta = 1`` recovers GBM and ``beta = 0`` an arithmetic model.
    """

    state_names = ("spot",)
    n_factors = 1

    def __init__(self, s0: float, mu: float, sigma: float, beta: float):
        check_positive(s0=s0)
        check_non_negative(sigma=sigma, beta=beta)
        self.s0 = float(s0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.beta = float(beta)

    def _simulate(self, grid, normals, rng):
        n_paths = normals.shape[0]
        s = np.empty((n_paths, grid.n_steps + 1))
        s[:, 0] = self.s0
        for i, dt in enumerate(grid.dt):
            si = s[:, i]
            alive = si > 0.0
            nxt = si + self.mu * si * dt + self.sigma * np.where(alive, si, 0.0) ** self.beta \
                * np.sqrt(dt) * normals[:, i, 0]
            s[:, i + 1] = np.where(alive, np.maximum(nxt, 0.0), 0.0)
        return {"spot": s}


class MultiAssetGBM(StochasticProcess):
    """Correlated geometric Brownian motions, simulated exactly.

    Parameters
    ----------
    s0, mu, sigma : array_like
        Initial values, drifts and volatilities, one per asset.
    correlation : array_like
        Correlation matrix of the Brownian drivers.
    q : float or array_like
        Dividend yields.
    """

    def __init__(self, s0, mu, sigma, correlation, q=0.0):
        s0 = np.atleast_1d(np.asarray(s0, dtype=float))
        n = s0.size
        self.s0 = s0
        self.mu = np.broadcast_to(np.asarray(mu, dtype=float), (n,)).copy()
        self.sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (n,)).copy()
        self.q = np.broadcast_to(np.asarray(q, dtype=float), (n,)).copy()
        if np.any(s0 <= 0) or np.any(self.sigma < 0):
            raise ValueError("s0 must be positive and sigma non-negative")
        self.correlation = np.asarray(correlation, dtype=float)
        self._chol = cholesky(self.correlation, n)
        self.n_factors = n
        self.state_names = tuple(f"asset_{i}" for i in range(n))

    def _simulate(self, grid, normals, rng):
        dt = grid.dt[None, :, None]
        z = normals @ self._chol.T
        inc = (self.mu - self.q - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
        n_paths = normals.shape[0]
        log_s = np.concatenate([np.zeros((n_paths, 1, self.n_factors)), np.cumsum(inc, axis=1)], axis=1)
        s = self.s0 * np.exp(log_s)
        return {name: s[:, :, i] for i, name in enumerate(self.state_names)}

    def mean(self, t):
        """Expected values, shape ``(..., n_assets)``."""
        t = np.asarray(t, dtype=float)[..., None]
        return self.s0 * np.exp((self.mu - self.q) * t)


class VarianceGamma(StochasticProcess):
    r"""Variance gamma model (Madan, Carr and Chang, 1998)

    .. math:: S_t = S_0 \exp\big((\mu + \omega) t + \theta G_t + \sigma W_{G_t}\big)

    where ``G`` is a gamma subordinator with unit mean rate and variance rate ``nu``, and
    ``omega = ln(1 - theta nu - sigma^2 nu / 2) / nu`` makes ``E[S_t] = s0 exp(mu t)``.
    Simulated exactly.
    """

    state_names = ("spot",)
    n_factors = 1

    def __init__(self, s0: float, mu: float, sigma: float, nu: float, theta: float):
        check_positive(s0=s0, nu=nu)
        check_non_negative(sigma=sigma)
        if 1.0 - theta * nu - 0.5 * sigma**2 * nu <= 0.0:
            raise ValueError("parameters must satisfy 1 - theta*nu - sigma^2*nu/2 > 0")
        self.s0 = float(s0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.nu = float(nu)
        self.theta = float(theta)

    @property
    def omega(self) -> float:
        return np.log(1.0 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu) / self.nu

    def _simulate(self, grid, normals, rng):
        dt = grid.dt
        n_paths = normals.shape[0]
        g = rng.gamma(shape=dt / self.nu, scale=self.nu, size=(n_paths, dt.size))
        inc = (self.mu + self.omega) * dt + self.theta * g + self.sigma * np.sqrt(g) * normals[:, :, 0]
        log_s = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)
        return {"spot": self.s0 * np.exp(log_s)}

    def mean(self, t):
        """Expected value ``E[S_t]``."""
        return self.s0 * np.exp(self.mu * np.asarray(t, dtype=float))
