# Changelog

## 0.3.0 — 2026-09-22

Speed release: training without MCMC, exact one-site conditionals, multi-chain sampling.

### Added
- **Pseudo-likelihood estimator** (`fit(method="pl")`, now the default): exact objective and gradient via 1-D
  quadrature, L-BFGS-B, optional L2 on W, respects `mu_global`, `lam_global`, `freeze`, `lam_min`. On synthetic
  φ⁴ data it recovers W with corr > 0.95; at V = 150, N = 80 it fits in ~15–20 s.
- **Exact conditionals**: `conditional_distribution`, `forecast_distribution` and the `ConditionalDistribution`
  class (mean, std, moments, pdf, cdf, quantile, interval, sample, affine change of units). Used automatically by
  `predict_conditional` / `forecast_next_day` (`method="auto"`) when only one site is free; a next-day forecast at
  V = 150 takes < 1 ms instead of ~0.1 s of MCMC.
- **Multi-chain samplers**: `MetropolisSampler(..., n_chains=C)` updates a site in all chains with one numpy step;
  new `HeatBathSampler` (exact Gibbs on a grid). `Phi4Model(sampler=..., n_chains=32)`.
- **Persistent chains** (PCD) in `fit(method="ml")`: burn-in only when a chain starts fresh.
- `l2=` penalty for `method="ml"` as well; `fit` continues from current couplings, so `pl` → `ml` refines.
- `scaling_analysis(method="pl")` (default) and `l2=`.
- `benchmarks/bench_v03.py`; 12 new tests (finite-difference gradient check, PL = exact log-density for one
  site, Metropolis vs heat-bath, exact vs MCMC conditionals, PL recovery, PL options).

### Changed
- `fit` defaults to `method="pl"`; pass `method="ml"` for the paper's estimator. `mcmc_steps` is now sweeps per
  chain per epoch (default 100).
- `sample` returns rows interleaved across chains; `last_state` has shape `(n_chains, V)`; `init` accepts
  `(V,)` or `(n_chains, V)`.
- Single-chain Metropolis uses a scalar loop (about 2× faster than 0.2.0); with 64 chains a chain-sweep at
  V = 150 costs about 1/15 of 0.2.0's (timings vary with machine load).

### Found while testing
- In the forecasting set-up (150 lags, 80 rows, 11,175 couplings) the fit overfits unless W is regularised
  strongly: on synthetic i.i.d. returns, `l2=1e-3` gave 37% coverage for the 90% interval and a worse MAE than the
  zero forecast; `l2=1` gave 88% and matched it. `examples/02_forecast.py` uses `l2=0.5`; choose it on validation
  data.

## 0.2.0 — 2026-09-22

Correctness release. Results produced with 0.1.0 should be discarded.

### Fixed
- **Gradient sign.** `fit` performed gradient descent on the log-likelihood. With the sign corrected the model
  recovers known couplings from synthetic φ⁴ data (corr(W) ≈ 0.96–0.98); 0.1.0 drove μ and λ strongly
  negative and learned W anti-correlated with the truth.
- **Conditional sampling.** Free sites that were not targets stayed at their random initial values for the whole
  chain; all free sites are now sampled, so they are marginalized.
- `n_samples` is honoured (was ignored); burn-in and thinning are parameters (burn-in was fixed at 1000).
- Conditional proposals use the model's `proposal_range` (were fixed at ±1.5).
- `batch_size` larger than the data set no longer crashes; full batch is the default.
- Scaling analysis freezes μ and λ for the sub-volume fits, uses the signed means of eq. 8, adds the full
  volume as a data point, fits the exponents with standard errors, and uses the paper's 100/40/40 iterations.

### Added
- `Scaler` (minmax, absmax, none): fitted on training data only, invertible, serializable.
- `scaler=` argument in `predict_conditional` / `forecast_next_day` to work in return units.
- `lag_embed` for the forecasting set-up.
- Training history (`model.history`): data vs model magnetization and susceptibility (Fig. 7), correlation gap,
  acceptance rate.
- `freeze=` and `lam_min=` options; per-model `numpy.random.Generator`.
- Metrics: `binarize`, `magnetization`, `susceptibility`; vectorized `market_kurtosis`.
- pytest suite and three runnable examples (multi-stock, forecast with zero baseline, imputation with baseline R).

### Changed
- `load_returns` returns raw returns; the `normalize` argument is removed (use `Scaler`).
- `MetropolisSampler.sample(n_samples, burn, thin, init, fixed)` returns full configurations and replaces
  `sample(n_steps, burn)`; `sample_conditional(fixed, ...)` takes a dict.
- Local O(V) action differences instead of full O(V²) recomputation per proposal.
- yfinance, matplotlib and tqdm are optional extras.
