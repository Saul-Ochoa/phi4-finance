"""Timing of the v0.3 engine at the paper's forecasting size (V = 150, N = 80).

Run: python benchmarks/bench_v03.py
Numbers depend on the machine; ratios are what matter.
"""
import time

import numpy as np

from phi4finance import MetropolisSampler, HeatBathSampler, Phi4Model, lag_embed

V, N = 150, 80
rng = np.random.default_rng(0)
series = np.tanh(rng.standard_t(4, N + V - 1) * 0.3)   # fat-tailed, inside [-1, 1]
X = lag_embed(series, V)

p = Phi4Model(V, seed=0)


def per_chain_sweep(cls, n_chains, sweeps=20):
    s = cls(p.W, p.mu, p.lam, p.a, n_chains=n_chains, rng=np.random.default_rng(0))
    t = time.perf_counter()
    s.sample(n_samples=n_chains * sweeps, burn=0)
    return (time.perf_counter() - t) / (sweeps * n_chains)


def timed(f):
    t = time.perf_counter(); out = f(); return out, time.perf_counter() - t


print(f"V = {V}, N = {N}")
base = per_chain_sweep(MetropolisSampler, 1)
print(f"Metropolis,   1 chain:           {base*1e3:8.3f} ms per chain-sweep")
for C in (16, 64, 256):
    t = per_chain_sweep(MetropolisSampler, C)
    print(f"Metropolis, {C:3d} chains:           {t*1e3:8.3f} ms per chain-sweep  ({base/t:5.1f}x)")
t = per_chain_sweep(HeatBathSampler, 64, sweeps=5)
print(f"Heat-bath,   64 chains:           {t*1e3:8.3f} ms per chain-sweep  ({base/t:5.1f}x)")

m_pl, t_pl = timed(lambda: Phi4Model(V, seed=0, mu_global=True).fit(X, method="pl", l2=1e-3, verbose=False))
print(f"\nfit(method='pl', l2=1e-3): {t_pl:6.1f} s  ({m_pl.fit_result_.nit} L-BFGS iterations)")
epochs, steps, C = 20, 30, 32
_, t_ml = timed(lambda: Phi4Model(V, seed=0, n_chains=C).fit(X, method="ml", epochs=epochs,
                                                            mcmc_steps=steps, verbose=False))
print(f"fit(method='ml', {epochs} epochs x {steps} sweeps x {C} chains): {t_ml:6.1f} s "
      f"-> 300 epochs ~ {t_ml/epochs*300/60:4.1f} min")

hist = X[-1, 1:]
d, t_exact = timed(lambda: m_pl.forecast_distribution(hist))
_, t_mcmc = timed(lambda: m_pl.forecast_next_day(hist, n_samples=3000, burn=200, method="mcmc"))
print(f"\nnext-day forecast, exact 1-D: {t_exact*1e3:7.2f} ms  (mean {d.mean():+.4f}, std {d.std():.4f})")
print(f"next-day forecast, MCMC 3000: {t_mcmc*1e3:7.0f} ms")
