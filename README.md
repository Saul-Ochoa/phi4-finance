# phi4-finance

Python implementation of the disordered φ⁴ quantum field theory for financial time series proposed in:

> D. Bachtis, D. S. Berman, A. Schelpe, *Modeling financial time series with φ⁴ quantum field theory*,
> Physica A 701 (2026) 132033. [doi:10.1016/j.physa.2026.132033](https://doi.org/10.1016/j.physa.2026.132033)

This is an independent implementation, not the authors' code.

## Why φ⁴ and not Ising?

The Ising model needs returns binarized to ±1, which destroys higher-order statistics such as the market
kurtosis, a possible crisis indicator. φ⁴ keeps continuous fields, is nonlinear, and every coupling stays
interpretable: `w_ij` couples two stocks, `a_i` biases one stock up or down, and `μ_i`, `λ_i` set the shape of
its distribution.

## Model

```
S(φ) = − Σ_ij w_ij φ_i φ_j + Σ_i μ_i φ_i² + Σ_i λ_i φ_i⁴ − Σ_i a_i φ_i,     p(φ) = exp(−S) / Z
```

Two estimators:

- `fit(X, method="pl")` (default): maximum pseudo-likelihood. Each one-site conditional is a 1-D density whose
  normaliser is computed by quadrature, so the objective and its gradient are exact and no MCMC is needed.
- `fit(X, method="ml")`: the paper's maximum likelihood (eq. 5), with persistent parallel Metropolis chains for the
  model expectations. `fit` continues from the current couplings, so `pl` then `ml` refines a PL solution.

Sampling runs `n_chains` chains at once (`sampler="metropolis"`, the paper's uniform proposal, or
`"heatbath"`, exact Gibbs). All samplers work on `[−proposal_range/2, proposal_range/2]` (default `[−1.5, 1.5]`,
as in the paper), so the distribution is `exp(−S)` **truncated to that hypercube**. Scale returns into roughly
`[−1, 1]` first.

When every site but one is known — the next-day forecast of Section 3.5, or imputing one stock from all the
others — `conditional_distribution` / `forecast_distribution` return that conditional exactly (mean, std,
quantiles, intervals, pdf, samples) without MCMC.

## Install

```bash
pip install -e .              # core: numpy, pandas, scipy
pip install -e ".[data]"      # + yfinance for load_returns
pip install -e ".[dev]"       # + pytest, matplotlib, tqdm, jupyter
```

## Quickstart

```python
from phi4finance import Phi4Model, Scaler, load_returns

rets = load_returns(["AAPL", "MSFT", "NVDA"], period="5y")   # return units
train = rets.iloc[:-20]                                        # keep the last 20 days out

scaler = Scaler("absmax").fit(train)                           # fit on training data only
model = Phi4Model(n_stocks=3, mu_global=False, lam_global=False)
model.fit(scaler.transform(train).to_numpy())                  # pseudo-likelihood

# p(NVDA | AAPL = +1%, MSFT = −0.5%): exact, in return units
d = model.conditional_distribution({0: 0.01, 1: -0.005}, 2, scaler=scaler)
print(d.mean(), d.std(), d.interval(0.9))

# p(NVDA, MSFT | AAPL = +1%): two free sites -> MCMC samples
s = model.predict_conditional({0: 0.01}, target_idx=[1, 2], scaler=scaler)
```

Next-day forecasting from a stock's own history (Section 3.5) uses `lag_embed` and `forecast_distribution`; see
`examples/02_forecast.py`. With 150 lags and 80 training rows the model has 11,175 couplings, so a strong L2
penalty (`l2=`) chosen on validation data is essential.

## Reproducing the paper

`notebooks/reproduce_bachtis2026.ipynb` reruns Fig. 1–6, 9 and 11 with public Yahoo Finance data, with the
paper's numbers next to the reproduced ones and extra baselines (zero forecast, OLS, ridge AR). See
`notebooks/README.md`.

## Structure

| Module | Contents |
| --- | --- |
| `phi4finance/model.py` | `Phi4Model`: training (`pl` / `ml`), sampling, conditional prediction, forecasting |
| `phi4finance/estimators.py` | pseudo-likelihood objective, exact gradient, L-BFGS fit |
| `phi4finance/inference.py` | `ConditionalDistribution`, exact one-site conditionals |
| `phi4finance/sampler.py` | `MetropolisSampler`, `HeatBathSampler`: multi-chain, local ΔS, clamped sites |
| `phi4finance/preprocessing.py` | `Scaler` (minmax, absmax), `lag_embed` |
| `phi4finance/data.py` | `load_prices`, `load_returns` (Yahoo Finance, optional; CSV cache) |
| `phi4finance/metrics.py` | market mean and kurtosis, binarization, SMA, sign agreement, MAE, coverage |
| `phi4finance/scaling.py` | finite-size scaling exponents `k_w`, `k_a` (Section 3.3) |
| `phi4finance/rolling.py` | `RollingPhi4`: one theory per date, warm-started, with sampled market statistics |
| `phi4finance/baselines.py` | baseline R (eq. 10), OLS, rolling AR, ridge AR |
| `phi4finance/validation.py` | `select_l2`: L2 penalty chosen on held-out rows |
| `notebooks/` | reproduction of the paper |
| `examples/` | 01 multi-stock fit, 02 next-day forecast, 03 imputation vs baseline R |
| `tests/` | pytest suite: recovery of known couplings (PL and ML), gradient checks, exact vs MCMC |
| `benchmarks/` | timing at the paper's forecasting size (V = 150, N = 80) |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Status

v0.4.0 adds the tools and the notebook to reproduce the paper (see `CHANGELOG.md`). Next (v0.5): Toeplitz couplings
for the time-lag model, stronger regularization and a full backtest module.
