# Backtesting risk-factor models

stochPathSim started as a tool for backtesting the risk-factor simulations behind counterparty
exposure models: simulate each risk factor over a horizon, then check where the realised value fell
in the simulated distribution.

- `pit_values` computes the probability integral transform of each realised value. A well-specified model gives uniform PITs, which `anderson_darling_uniform` and `cramer_von_mises_uniform` test.
- `kupiec_pof` tests the number of quantile exceedances, and `christoffersen_independence` tests whether they cluster.
- `path_statistics` and `exceedance_counts` give per-date quantile bands and how often a realised path breached them.

```python
--8<-- "examples/04_risk_factor_backtest.py"
```
