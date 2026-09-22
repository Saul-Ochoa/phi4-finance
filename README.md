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

The couplings are learned by maximum likelihood (eq. 5 of the paper) with Metropolis estimates of the model
expectations. Proposals are uniform on `[−proposal_range/2, proposal_range/2]` (default `[−1.5, 1.5]`, as in
the paper), so the sampled distribution is `exp(−S)` **truncated to that hypercube**. Scale returns into
roughly `[−1, 1]` first.

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
model = Phi4Model(n_stocks=3, lr=0.01, mu_global=False, lam_global=False)
model.fit(scaler.transform(train).to_numpy(), epochs=400, mcmc_steps=400)

# p(NVDA | AAPL = +1%, MSFT = −0.5%), inputs and samples in return units
samples = model.predict_conditional({0: 0.01, 1: -0.005}, target_idx=2, scaler=scaler)
print(samples.mean(), samples.std())
```

Next-day forecasting from a stock's own history (Section 3.5) uses `lag_embed` and `forecast_next_day`; see
`examples/02_forecast.py`.

## Structure

| Module | Contents |
| --- | --- |
| `phi4finance/model.py` | `Phi4Model`: training, sampling, conditional prediction, forecasting |
| `phi4finance/sampler.py` | `MetropolisSampler` with local ΔS and conditional (clamped) sampling |
| `phi4finance/preprocessing.py` | `Scaler` (minmax, absmax), `lag_embed` |
| `phi4finance/data.py` | `load_returns` (Yahoo Finance, optional) |
| `phi4finance/metrics.py` | market mean and kurtosis, binarization, SMA, magnetization, susceptibility |
| `phi4finance/scaling.py` | finite-size scaling exponents `k_w`, `k_a` (Section 3.3) |
| `examples/` | 01 multi-stock fit, 02 next-day forecast, 03 imputation vs baseline R |
| `tests/` | pytest suite, including recovery of known couplings from synthetic φ⁴ data |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Status

v0.2.0 fixes the correctness issues of v0.1 (see `CHANGELOG.md`). Next: a vectorized sampler,
pseudo-likelihood training and exact one-site conditionals (v0.3), then reproduction of the paper's figures.
