# Changelog

## 0.5.4 — 2026-09-24

### Documentation
- README rewritten (in Spanish): what the library is for, what it is not for, quick start, modules.
- `docs/metodologia.md`, `docs/proceso.md`, `docs/conclusiones.md`: model and evaluation method, development history,
  and the results on real data with the final verdict.
- `LICENSE` (MIT).

### Changed
- Only `notebooks/validacion_riesgo_sistemico.ipynb` is published, with the outputs of the 23-sep-2026 run. The
  reproduction, Magnificent 7 and risk-map notebooks mentioned in earlier entries are kept out of the repository.
- The validation notebook no longer uses `DataFrame.query` (it failed on Python 3.13 with column names such as
  "R² oos").

## 0.5.3 — 2026-09-23

### Changed
- `validacion_riesgo_sistemico.ipynb`: horizons of 20 and 60 days, SPY as a second target next to the equal-weight
  portfolio, and the standardised shifts of Kritzman et al. (2011) for the absorption ratio, the average correlation
  and the φ⁴ coupling. Base B now also includes Δ absorption ratio. Results are summarised as heat maps over the 8
  asset × horizon × target combinations; a cell is starred only when R² oos > 0 and the Clark–West p < 0.05.

## 0.5.2 — 2026-09-23

### Added
- `phi4finance.earlywarning`: benchmark indicators (`average_correlation`, `absorption_ratio`), forward targets
  (`forward_realized_vol`, `forward_max_drawdown`), `hac_ols` (Newey–West), `oos_r2` (expanding-window
  out-of-sample R² with a gap for overlapping targets and a Clark–West p-value) and `auc`. 5 new tests (61 total).
- `RollingPhi4` records scale-free indicators: `coupling_mean`, `coupling_absmean`, `hub_strength_max`, `log_mu_std`.
- `notebooks/validacion_riesgo_sistemico.ipynb`: do φ⁴ indicators anticipate the next 20 days' volatility and
  drawdown of the top-20 S&P 500 portfolio beyond current volatility, average correlation and the absorption ratio?

## 0.5.1 — 2026-09-23

### Added
- `phi4finance.risk`: `coupling_matrix` (scale-free couplings w_ij/√(μ_i μ_j)), `node_strength`,
  `stress_matrix` (E[φ_j | φ_i = shock] by MCMC for every pair), `empirical_stress` (the same on the data's tail
  days) and `var_es`. 3 new tests (56 total).
- `notebooks/sp500_top20_riesgo.ipynb`: risk map of the 20 largest S&P 500 stocks over 10 years, 3 years and the
  current year.

## 0.5.0 — 2026-09-23

Changes proposed after running the notebooks on real data (Magnificent 7, Mar–Sep 2026).

### Added
- **Scale-free L2 penalty** (`fit(..., penalty_scale="std")`, default): l2 · Σ (s_i s_j w_ij)² with s = the data's
  standard deviation, so the penalty acts on standardised couplings and does not depend on how the data were scaled.
  `penalty_scale="none"` restores the 0.4 penalty. The default `select_l2` grid is now decades from 0.01 to 10⁴.
- **Parameter tying** (`phi4finance.structure`): `Tying.free`, `Tying.toeplitz` (one stock's lags,
  w_ij = w(|i − j|)), `Tying.lagged` (several stocks over several days, block-Toeplitz with lead–lag direction),
  `lag_embed_panel`. `Phi4Model(tying=...)`; pseudo-likelihood and ML both fit one parameter per group.
- **Volatility filters** (`phi4finance.volatility`): `EWMAVol` (RiskMetrics), `GARCHVol` (GARCH(1,1) QMLE),
  `devolatilize`; one-step-ahead σ_{t|t−1}, no look-ahead.
- **Distributional scores**: `ConditionalDistribution.crps / logpdf / pit / gaussian`, `metrics.crps_gaussian`,
  `metrics.diebold_mariano` (Newey–West).
- **Backtest** (`phi4finance.backtest`): `walk_forward`, `summarize`, `GaussianVolForecaster` (N(0, σ²) with
  EWMA or GARCH) and `Phi4LaggedForecaster` (φ⁴ on r/σ over n lags of one or all stocks, L2 by blocked CV on CRPS,
  biases fixed at 0 by default).
- `select_l2(folds=k, metric="crps")`: blocked cross-validation and CRPS as the selection metric.
- A warning when `lam_min < 0` is used with pseudo-likelihood.
- 12 new tests (53 total).

### Changed
- Notebooks: loader prints only the library's own notes (no yfinance "unclosed database" noise).
- `mag7_ultimos_6_meses.ipynb`: §2 without penalty and in partial-correlation units; §4 adds φ⁴ on EWMA-standardised
  returns; §5 chooses L2 by blocked CV with a = 0; new §6 runs the v0.5 backtest.
- `reproduce_bachtis2026.ipynb`: §3.5 chooses L2 by blocked CV.

### Results on real data (Magnificent 7, 127 days to 22 Sep 2026)
- Structure: without the penalty, the model's implied partial correlations w_ij/√(μ_i μ_j) equal the sample ones
  (r = 1.00; signs agree on 100% of pairs). The 0.4 notebook's fixed L2 had made the signs follow plain correlations.
- Imputation: with λ at its floor φ⁴ equals OLS exactly. The EWMA filter lifts 90%-interval coverage from 86% to
  91%; days with idiosyncratic jumps (earnings) are still missed by every method.
- Next-day forecasts: GARCH beats EWMA on CRPS (ratio 0.989, DM p < 0.001). φ⁴ with 5 lags of all stocks: 1.006,
  own lags only: 1.009 (both significantly worse on CRPS, better coverage and log score); cross-validation pushed
  L2 to the top of the grid for most stocks, i.e. the lags carry no usable signal in this period. The paper's
  150-lag set-up went from φ⁴/zero = 1.007 to 1.003 (p = 0.60) with blocked CV and a = 0.

## 0.4.0 — 2026-09-22

Reproduction release: everything needed to rerun the paper with public data.

### Added
- `notebooks/reproduce_bachtis2026.ipynb`: §3.1–3.5 (Fig. 1–6, 9, 11) with Yahoo Finance data, a `FULL` switch
  for paper-scale settings, the paper's numbers beside the reproduced ones, and every deviation stated. Committed
  without outputs.
- `RollingPhi4`: one theory per date on a rolling window, warm-started, recording coupling summaries and data vs
  model market mean and kurtosis (Fig. 1).
- `phi4finance.baselines`: rescaled mean R (eq. 10), OLS, rolling AR (the paper's linear-regression baseline),
  ridge AR on the same lags as φ⁴.
- `phi4finance.validation.select_l2`: L2 chosen on held-out rows, warm-started from strong to weak penalty.
- Metrics: `sign_product_matrix`, `sign_agreement`, `mae`, `mae_se`, `coverage`, `hit_rate`.
- `load_prices` with a CSV cache (`cache_dir`), `adjusted=` switch and removal of tickers with short history;
  `load_returns` forwards them.
- `ScalingResult.exponents(statistic)`: signed and absolute means from one run; `scaling_analysis(n_grid=)`.
- 5 new tests (41 total).

### Changed
- Pseudo-likelihood objective about 2× faster (analytic upper bound instead of a max-reduction over the grid,
  in-place exponentials, exact fallback for extreme couplings).

### Notes from building the notebook
- sgn(w_ij) should be compared with **partial** correlations: in the Gaussian limit the precision matrix is
  2(diag μ − W). On synthetic one-factor data, sgn(w) matched the partial-correlation signs on 100% of pairs and the
  plain-correlation signs on 73%.
- k_w ≈ −1 is what a one-factor market predicts (precision off-diagonals shrink as 1/V). The notebook adds this
  null and the Gaussian limit as benchmarks for the paper's exponents.

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
