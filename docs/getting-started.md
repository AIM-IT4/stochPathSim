# Getting started

## Installation

```bash
pip install stochPathSim              # numpy, scipy, pandas, plotly
pip install "stochPathSim[export]"    # adds kaleido for static PNG/SVG export
```

## Simulate, inspect, price

Every process has the same `simulate` method and returns a `SimulationResult`:

```python
import stochPathSim as sps

model = sps.Heston(s0=100, mu=0.03, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
paths = model.simulate(T=1.0, n_steps=252, n_paths=50_000, seed=42)

paths.names                   # ('spot', 'variance')
paths["spot"].shape           # (50000, 253): column 0 is the initial value
paths.terminal()              # values of the primary state at T
paths.quantile([0.01, 0.99])  # quantile bands at every time
paths.to_frame(max_paths=5)   # pandas DataFrame indexed by time
paths.plot()                  # interactive Plotly chart: paths, mean and 1%-99% band
paths.fan()                   # nested quantile bands
paths.distribution(0.5)       # histogram of the values at t = 0.5
```

Common options:

- `seed=` for reproducible results
- `antithetic=True` for antithetic variates
- `grid=sps.TimeGrid([...])` for non-uniform time grids
- `normals=` to drive several models with the same random numbers

## A complete example

```python
--8<-- "examples/01_heston_pricing.py"
```
