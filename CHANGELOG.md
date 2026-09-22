# Changelog

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
