# stochPathSim

**Stochastic path simulation, derivatives pricing and risk analytics for quantitative finance.**

<iframe src="assets/stochpathsim-paths.html" width="100%" height="460" style="border:none"></iframe>

*Interactive: hover, zoom and toggle series.*

stochPathSim simulates the stochastic processes used in equity, FX, commodity, credit and interest-rate modelling, prices
derivatives on the simulated paths, and turns them into risk measures: exposure profiles, CVA and VaR.
It also includes the statistical tests used to backtest risk-factor models.

Every model is validated in the test suite against an independent reference: Monte Carlo prices are
checked against closed forms within a few standard errors, and the semi-analytic formulas (Heston, SABR,
Hull-White bond options) agree with QuantLib to 1e-10 or better.

| Package | Contents |
| :-- | :-- |
| `stochPathSim.processes` | GBM, multi-asset GBM, Merton, Heston, Bates, SABR, CEV, variance gamma, OU, Vasicek, CIR, Hull-White, Ho-Lee, G2++, Black-Karasinski, BDT |
| `stochPathSim.analytics` | Black-Scholes, Black-76, Bachelier, Heston, Merton, SABR, bond prices and bond options |
| `stochPathSim.pricing` | payoffs, Monte Carlo pricer, Longstaff-Schwartz, binomial tree |
| `stochPathSim.risk` | exposure (EE, PFE, EPE, Effective EPE), CVA, VaR/ES, backtesting tests |
| `stochPathSim.validation` | convergence and discretisation studies, model comparison, HTML validation reports |
| `stochPathSim.viz` | interactive Plotly charts and dashboards |
| `stochPathSim.calibration` | maximum-likelihood fits for GBM, OU/Vasicek and CIR |

```bash
pip install stochPathSim
```

Continue with [Getting started](getting-started.md).
