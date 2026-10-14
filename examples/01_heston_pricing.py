"""Price a European call under Heston three ways: semi-analytic, Monte Carlo and Black-Scholes."""

import stochPathSim as sps
from stochPathSim.analytics import bs_implied_vol, heston_price
from stochPathSim.pricing import EuropeanCall, price_mc

params = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
S, K, T, r = 100.0, 110.0, 1.0, 0.03

exact = heston_price(S, K, T, r, 0.0, **params, option_type="call")
model = sps.Heston(s0=S, mu=r, **params)
mc = price_mc(model, EuropeanCall(K), T, discount=r, n_steps=50, n_paths=200_000, seed=42, antithetic=True)

print(f"Heston semi-analytic : {exact:.4f}")
lo, hi = mc.conf_int(0.95)
print(f"Heston Monte Carlo   : {mc.price:.4f} +/- {mc.stderr:.4f}  (95% CI {lo:.4f} to {hi:.4f})")
print(f"Implied Black vol    : {bs_implied_vol(exact, S, K, T, r):.4%}")
