# Pricing derivatives

`price_mc(process, payoff, T, ...)` simulates the process, evaluates the payoff on every path and
discounts it. It returns an `MCResult` with the price, its standard error and `conf_int(level)`.

Discounting can be:

- a flat continuously compounded rate (`discount=0.03`)
- a `ZeroCurve`
- `None`, which uses the process's own pathwise `discount` state for short-rate models

```python
import stochPathSim as sps
from stochPathSim.pricing import AsianArithmetic, AsianGeometric, price_mc

model = sps.GBM(100, 0.05, 0.2)
plain = price_mc(model, AsianArithmetic(100), 1.0, discount=0.05, n_steps=12, n_paths=100_000, seed=1)

# geometric Asian as a control variate (its exact price is known)
geo_exact = price_mc(model, AsianGeometric(100), 1.0, discount=0.05, n_steps=12, n_paths=2_000_000, seed=2).price
cv = price_mc(model, AsianArithmetic(100), 1.0, discount=0.05, n_steps=12, n_paths=100_000, seed=1,
              control_variate=(AsianGeometric(100), geo_exact))
print(plain.stderr, cv.stderr)   # the control variate cuts the error by an order of magnitude
```

American options use least-squares Monte Carlo:

```python
import numpy as np
from stochPathSim.pricing import crr_price, longstaff_schwartz

lsm = longstaff_schwartz(sps.GBM(36, 0.06, 0.2), lambda s: np.maximum(40 - s, 0), 1.0,
                         r=0.06, n_steps=50, n_paths=100_000, seed=3)
tree = crr_price(36, 40, 1.0, 0.06, 0.2, "put", american=True)
```
