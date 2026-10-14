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
