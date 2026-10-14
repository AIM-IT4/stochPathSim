# Changelog

All notable changes to this project are documented here. The project follows [Semantic Versioning](https://semver.org).

## [2.0.0] - 2026-10-14

A rewrite into a validated simulation, pricing and risk library. The 1.x interface keeps working.

### Added
- **Core:**
  - `TimeGrid` with uniform, refined and arbitrary grids.
  - `SimulationResult` with quantiles, plotting and DataFrame export.
  - `StochasticProcess` base class with seeds, antithetic variates and common random numbers.
- **`ZeroCurve`:** monotone-cubic or linear interpolation, discount factors and forward rates.
- **Processes:**
  - Equity and generic: `GBM` (exact), `MultiAssetGBM`, `MertonJumpDiffusion`, `Heston` (Andersen QE and full-truncation Euler), `Bates`, `SABR`, `CEV`, `VarianceGamma`, `OrnsteinUhlenbeck`.
  - Rates: `Vasicek` and `CIR` (exact schemes), curve-fitted `HullWhite`, `HoLee` and `G2pp`, `BlackKarasinski`, `BlackDermanToy`.
  - Short-rate models return pathwise discount factors.
- **Analytics:**
  - Black-Scholes prices, Greeks and implied vol; Black-76; Bachelier price and implied vol.
  - Heston semi-analytic price; Merton series price; Hagan SABR (including shifted).
  - Vasicek, CIR, Hull-White and G2++ bond prices; Vasicek and Hull-White bond options; Hull-White caplets.
- **Pricing:**
  - Payoffs: European, digital, Asian, barrier, lookback, basket and spread.
  - `price_mc` with standard errors, antithetic and control variates.
  - Longstaff-Schwartz American pricing and a CRR binomial tree.
- **Risk:**
  - Exposure profiles (EE, PFE, EPE, Effective EE/EPE), collateralised exposure and CVA.
  - VaR and ES, path statistics and exceedance counts.
  - PIT, Anderson-Darling, Cramér-von Mises, Kupiec and Christoffersen backtests.
- **Calibration:** `fit_gbm`, `fit_ou` (Vasicek) and `fit_cir` maximum-likelihood estimators.
- **Project:** test suite validated against closed forms and QuantLib, CI on Linux, macOS and Windows (Python 3.10–3.13), documentation site, examples and `CITATION.cff`.

### Fixed (1.x behaviour)
- **GBM:** paths now start at `s0`. 1.x applied a Brownian increment at `t = 0`.
- **Heston:** 1.x had no drift and used the next step's variance for the current step.
- **Hull-White:** 1.x simulated an Ornstein-Uhlenbeck process reverting to zero. `HullWhiteModel` is now a Hull-White model fitted to a flat curve at `r0`.
- **Black-Derman-Toy:** 1.x simulated an arithmetic random walk. The rate is now lognormal.
- **Plotting:** `plot_simulation` no longer asks for keyboard input and returns the Axes.
- **Statistics:** `display_statistics` no longer modifies its argument.
- **Dependencies:** `scipy` is now declared, and the 1.x tests (which imported a package that no longer existed) are replaced.

### Changed
- Python 3.10+ is required. matplotlib is optional (`pip install "stochPathSim[plot]"`).
- 1.x classes accept an optional `seed` and use NumPy's `Generator` instead of the global random state.

## [1.1.1] - 2024-12-15
- Initial public release on PyPI.
