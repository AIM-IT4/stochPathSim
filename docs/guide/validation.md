# Validation lab

Simulating paths is the easy part. The validation lab checks whether the numbers can be trusted.

## Convergence

`convergence_study` prices a payoff with an increasing number of paths and compares each price with a
reference. An unbiased estimator stays within a few standard errors (`z_score`), and its error falls
like 1/sqrt(N).

## Discretisation bias

`discretisation_study` reprices on the same random stream with more and more time steps. The bias shows
whether a scheme is accurate on a coarse grid, and the log-log slope of |bias| against dt estimates its
weak order. For Heston, the full-truncation Euler scheme gives an order close to 1, while Andersen's QE
scheme is unbiased within noise even at a handful of steps.

## Model comparison

`compare_models` simulates several models and tabulates terminal moments and quantiles, with an
overlaid density chart.

## Reports

`ValidationReport` combines all of the above, plus risk-factor backtests (PIT, Anderson-Darling, Kupiec,
Christoffersen), into one self-contained HTML file with pass/fail badges:

```python
--8<-- "examples/05_validation_report.py"
```

![A validation report](../assets/validation-report.png)
