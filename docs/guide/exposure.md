# Exposure and CVA

Given simulated mark-to-market values of a trade or netting set (`n_paths x n_times`),
`ExposureProfile.from_mtm` computes:

- expected exposure (EE) and expected negative exposure (ENE)
- PFE at a chosen quantile
- EPE, Effective EE and Effective EPE, following the Basel definitions

`collateralised_exposure` applies variation margin with a threshold, a minimum transfer amount and a
margin period of risk. `cva_from_exposure` turns an EE profile into a unilateral CVA.

```python
--8<-- "examples/03_exposure_profile.py"
```
