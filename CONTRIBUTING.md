# Contributing to stochPathSim

Thanks for helping improve stochPathSim! Bug reports, new models, payoffs and documentation fixes are all welcome.

## Ground rules

- **Numbers must be checked.** Every model or pricer comes with a test that compares it with an
  independent reference: a closed form, a known limit, a published table or another library
  (QuantLib, FinancePy). Monte Carlo tests use a fixed seed and a tolerance of a few standard errors.
- Keep pull requests focused on one change, and describe the problem and the fix in the PR.
- Discuss large or breaking changes in an issue first.
- By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

```bash
git clone https://github.com/AIM-IT4/stochPathSim.git
cd stochPathSim
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest
ruff check .
```

## Adding a model

1. Subclass `stochPathSim.core.StochasticProcess`, set `state_names` and `n_factors`, and implement
   `_simulate(grid, normals, rng)` returning one `(n_paths, n_steps + 1)` array per state, with
   column 0 equal to the initial state. Prefer exact transitions over Euler where they exist.
2. Validate parameters in `__init__` and document the SDE in the class docstring.
3. Add analytic moments (`mean`, `variance`) or prices where they exist, and test the simulation against them.
4. Export the class from the subpackage `__init__.py`, and add it to the README model table and `CHANGELOG.md`.

## Reporting bugs

Open an issue with a minimal example, the value you got, the value you expected and where the
expected value comes from.
