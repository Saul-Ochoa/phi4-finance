# Notebooks

| Notebook | What it does |
| --- | --- |
| `reproduce_bachtis2026.ipynb` | Reruns every empirical result of Bachtis, Berman & Schelpe (2026) — Fig. 1–6, 9 and 11 — with public Yahoo Finance data |
| `mag7_ultimos_6_meses.ipynb` | (Spanish) The Magnificent 7 over the 6 months up to the run date: learned structure, rolling couplings, same-day imputation and next-day forecasts, all out of sample against OLS / zero / ridge / AR(1) baselines, plus the v0.5 distributional backtest (volatility
filters, cross-asset lags, CRPS); exports a summary CSV to `results/` |

## Run

```bash
pip install -e ".[notebooks]"      # from the repository root
cd notebooks
jupyter lab                         # open either notebook
```

- The first run downloads prices from Yahoo Finance and caches them as CSV files in `notebooks/data/`
  (ignored by git). Later runs read the cache; delete the folder to re-download.
- `FULL = False` (default) runs reduced settings; `FULL = True` uses the paper's iteration counts and one fit per
  day in §3.1, which takes hours.
- `METHOD = "pl"` (default) uses pseudo-likelihood; `METHOD = "ml"` uses the paper's MCMC maximum likelihood and is
  much slower.

## What differs from the paper

The notebook lists every deviation at the top and in each section. In short: Yahoo adjusted closes instead of
WRDS data, pseudo-likelihood instead of MCMC likelihood by default, L2 chosen on 10 held-out rows instead of early
stopping, an assumed list of 64 stocks for the scaling analysis, no LSTM baseline, and extra baselines (zero,
training mean, OLS, ridge AR) plus two no-fit benchmarks for the scaling exponents.

The notebook is committed without outputs: every number in it is produced by your own run.
