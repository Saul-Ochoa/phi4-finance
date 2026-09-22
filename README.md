
# phi4-finance

First open-source implementation of phi^4 QFT for financial time series.

Paper: Bachtis et al. Modeling financial time series with phi^4 QFT, Physica A 701 (2026)

## Why phi^4 not Ising?
Ising binarizes returns -> destroys kurtosis (crisis indicator). phi^4 is continuous, nonlinear, interpretable.

## Install
pip install -e .
pip install -e .[torch]

## Quickstart
from phi4finance.data import load_returns
from phi4finance.model import Phi4Model

rets = load_returns(['AAPL','MSFT','NVDA'], period='5y')
model = Phi4Model(n_stocks=rets.shape[1])
model.fit(rets.values, epochs=500, mcmc_steps=2000)
pred = model.predict_conditional({0: 0.01, 1: -0.005}, target_idx=2)

## Structure
phi4finance/model.py - core
phi4finance/sampler.py - Metropolis
phi4finance/data.py - yfinance loader
phi4finance/metrics.py - market mean/kurtosis
phi4finance/scaling.py - scaling exponents
