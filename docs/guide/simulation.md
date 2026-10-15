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
