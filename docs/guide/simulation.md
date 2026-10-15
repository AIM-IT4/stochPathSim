# Simulating processes

Prefer exact transitions where they exist: GBM, OU, Vasicek, CIR (non-central χ²), Hull-White, Ho-Lee
and G2++ are all simulated without discretisation error, so a coarse grid is enough when you only need
values on a few dates. Heston uses Andersen's quadratic-exponential scheme by default.

## Interest rates fitted to a curve

Short-rate models return the short rate and the pathwise discount factor `exp(-∫ r dt)`.
Hull-White, Ho-Lee and G2++ are fitted to a `ZeroCurve`, so their Monte Carlo bond prices reproduce it:

```python
--8<-- "examples/02_hull_white_curve.py"
```

`ZeroCurve` interpolates zero rates with a monotone cubic by default, which keeps instantaneous forward
rates continuous: curve-fitted models follow the forward curve, so a linear interpolation would show
kinks at the pillars.

## Several assets

`MultiAssetGBM` takes a correlation matrix and returns one state per asset (`asset_0`, `asset_1`, …),
which `Basket` and `Spread` payoffs read directly.


## Normal forwards and NIG equity returns

`ArithmeticBrownianMotion(x0, mu, sigma)` simulates `dX = mu dt + sigma dW`
exactly, including negative values. Drift and volatility are in **absolute**
units. To price a normal forward, set `mu=0` under the forward measure and
supply the discount rate separately to `price_mc`. Compare with
`bachelier_price`; `bachelier_greeks` returns forward delta/gamma, vega per unit
normal volatility and theta holding the forward and discount factor fixed.
It is not a risk-neutral spot model obtained by setting `mu=r`.

`NormalInverseGaussian(s0, mu, alpha, beta, delta, q=0)` models positive equity
prices with skewed, heavier-tailed log returns. `alpha` controls tail decay,
`beta` asymmetry and `delta` activity per year. Compensation makes the expected
spot `s0*exp((mu-q)*t)`. Use `mu=r` for risk-neutral pricing. Exact inverse
Gaussian clock increments work on arbitrary time grids.

The constructor requires `alpha > abs(beta)` and `alpha > abs(beta+1)`.
`variance(t)` additionally requires `alpha > abs(beta+2)` when activity is
positive; a valid first moment does not guarantee a finite variance. Setting
`delta=0` gives deterministic growth. `log_return_mean` and
`log_return_variance` describe log returns, not spot prices.

`nig_price` prices scalar European calls/puts using SciPy NIG probabilities
under the original and stock measures. These are numerically evaluated
probabilities, not Monte Carlo estimates. Parameters follow
[SciPy's NIG parameterisation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norminvgauss.html)
with `a=alpha*delta*t`, `b=beta*delta*t`, `scale=delta*t` for the unshifted Levy increment.

With NIG, `normals=` fixes only the Gaussian shocks; reuse the seed to reproduce
clock draws too. Antithetic sampling pairs only the Gaussian shocks and does
not guarantee a variance reduction for every payoff.

Run the following example to compare both pricing methods and save an
interactive chart to `examples/output/normal-and-nig.html`:

```python
--8<-- "examples/06_normal_and_nig.py"
```

## Displaced diffusion and CIR++

Use `DisplacedDiffusion(s0, mu, sigma, shift, q=0)` for a forward whose value
can be negative, but whose shifted value `S + shift` is positive. The
shifted value is simulated exactly as a GBM. For a forward under its
forward measure set `mu=q=0`, so its expectation is the input forward.
Call `displaced_diffusion_price(F, K, T, df, sigma, shift)` to price options
using shifted Black-76, including negative strikes; the matching
`displaced_diffusion_greeks` gives forward delta/gamma, sigma-vega and
theta at fixed forward and discount factor. Drift acts on `S + shift`,
not on `S` itself.

`CIRPlusPlus(x0, kappa, theta, sigma, curve)` fits any `ZeroCurve` by
adding the deterministic difference between market and CIR instantaneous
forwards to an exact CIR factor. It returns `rate`, `factor` and `discount`;
the positive CIR factor does not guarantee positive shifted short rates.
The deterministic discount correction is exact, but the factor's integrated
discount is trapezoidal, so simulated bond prices have time-step bias.
Refine `n_steps` when accuracy matters. `scheme='euler'` allows common
Gaussian draws, while the default exact factor draws additional random
variates from the seed and ignores the supplied normals.

A runnable example with interactive Plotly charts:

```python
--8<-- "examples/07_displaced_and_cirpp.py"
```

## Advanced stochastic models and variance reduction

GARCH(1,1) and GJR-GARCH(1,1) represent volatility clustering and leverage
effects as *per-observation* conditional variance recursions. They are
discrete-time statistical models, not risk-neutral pricing models. Do not
interpret them as continuous-time diffusions or compare prices across
different numbers of time steps without recalibrating the parameters.

```python
import stochPathSim as sps
model = sps.GJRGARCH11(100, 0.03, omega=0.002, alpha=0.05, beta=0.85, gamma=0.08)
result = model.simulate(T=1, n_steps=252, n_paths=1024, seed=42)
result.plot(name="variance").show()
```

`FractionalBrownianMotion` uses a Cholesky factorisation of the exact
increment covariance for any grid; `hurst=0.5` is standard Brownian motion.
For other Hurst exponents it is not a semimartingale: do not use it as a
risk-neutral traded underlying. The O(n_steps^3) covariance factorisation
is intended for grids with a few hundred observations.

For Gaussian-factor Monte Carlo use scrambled Sobol quasi-random normals,
optionally ordered by a Brownian bridge so early coordinates determine
coarse-scale moves. Use independent scrambled replications to estimate
QMC uncertainty; ordinary iid standard errors computed across Sobol
points are not valid uncertainty estimates.

```python
from stochPathSim.core import sobol_normals
z = sobol_normals(4096, 32, seed=7, brownian_bridge=True)
paths = sps.GBM(100, 0.03, 0.2).simulate(T=1, n_steps=32, normals=z)
paths.fan().show()
```

The source example demonstrates all the new capabilities:

```python
--8<-- "examples/08_advanced_models.py"
```
