"""A phi^4 theory per point in time (Section 2: "a different field theory ...
for each point in time and the time evolution of the financial system is
represented by a time evolution in model space").

``RollingPhi4`` fits one model per rolling window, warm-starting each fit from
the previous couplings, and records for every window:

* the couplings' summaries (<w_ij>, <|w_ij|>, <a_i>, mu, lambda) and their
  scale-free versions (mean of w_ij / sqrt(mu_i mu_j), strongest hub, spread
  of log mu_i across assets);
* market statistics of the data in the window (mean over days of the
  cross-sectional mean and kurtosis);
* the same statistics computed on configurations sampled from the fitted
  model, back in return units (Fig. 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import market_kurtosis, market_mean
from .model import Phi4Model
from .preprocessing import Scaler

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


class RollingPhi4:
    """Rolling-window phi^4 fits.

    Parameters
    ----------
    window : rows (trading days) per fit.
    step : rows between consecutive fit dates.
    scaler : ``Scaler`` method fitted on each window ("absmax" or "minmax").
    method, l2 : estimator settings passed to ``Phi4Model.fit``.
    n_samples, burn : model configurations drawn per window (0 skips sampling).
    warm_start : start each fit from the previous window's couplings.
    keep_models : keep every fitted model in ``models_`` (memory grows with
        the number of windows); the last one is always in ``last_model_``.
    model_kw, fit_kw : extra arguments for ``Phi4Model`` and ``fit``.
    """

    def __init__(self, window: int = 250, step: int = 20, scaler: str = "absmax",
                 method: str = "pl", l2: float = 0.0, n_samples: int = 2000, burn: int = 200,
                 warm_start: bool = True, keep_models: bool = False, model_kw=None,
                 fit_kw=None, seed: int = 0, verbose: bool = True):
        self.window, self.step = int(window), int(step)
        self.scaler, self.method, self.l2 = scaler, method, float(l2)
        self.n_samples, self.burn = int(n_samples), int(burn)
        self.warm_start, self.keep_models = warm_start, keep_models
        self.model_kw = dict(model_kw or {})
        self.fit_kw = dict(fit_kw or {})
        self.seed, self.verbose = seed, verbose

    def fit(self, returns: pd.DataFrame) -> "RollingPhi4":
        """``returns``: (T, V) DataFrame of returns in return units, date index."""
        if not isinstance(returns, pd.DataFrame):
            returns = pd.DataFrame(returns)
        T, V = returns.shape
        if T < self.window:
            raise ValueError(f"need at least window={self.window} rows, got {T}")
        ends = list(range(self.window, T + 1, self.step))
        if ends[-1] != T:
            ends.append(T)
        it = _tqdm(ends, desc="windows") if (self.verbose and _tqdm is not None) else ends
        iu = np.triu_indices(V, 1)
        rows, self.models_, prev = [], [], None
        for k, end in enumerate(it):
            win = returns.iloc[end - self.window:end]
            sc = Scaler(self.scaler).fit(win)
            m = Phi4Model(V, seed=self.seed + k, **self.model_kw)
            if self.warm_start and prev is not None:
                m.W, m.a, m.mu, m.lam = (prev.W.copy(), prev.a.copy(), prev.mu.copy(), prev.lam.copy())
            m.fit(sc.transform(win).to_numpy(), method=self.method, l2=self.l2, verbose=False,
                  **self.fit_kw)
            rec = {"date": win.index[-1],
                   "w_mean": float(m.W[iu].mean()), "w_absmean": float(np.abs(m.W[iu]).mean()),
                   "a_mean": float(m.a.mean()), "mu_mean": float(m.mu.mean()),
                   "lam_mean": float(m.lam.mean()),
                   "data_market_mean": float(market_mean(win).mean()),
                   "data_market_kurtosis": float(np.nanmean(market_kurtosis(win)))}
            if np.all(m.mu > 0):                  # scale-free couplings w_ij / sqrt(mu_i mu_j)
                Cf = m.W / np.sqrt(np.outer(m.mu, m.mu))
                strength = np.abs(Cf).sum(axis=1) - np.abs(np.diag(Cf))
                rec.update({"coupling_mean": float(Cf[iu].mean()), "coupling_absmean": float(np.abs(Cf[iu]).mean()),
                            "hub_strength_max": float(strength.max()), "log_mu_std": float(np.log(m.mu).std())})
            else:
                rec.update({"coupling_mean": np.nan, "coupling_absmean": np.nan,
                            "hub_strength_max": np.nan, "log_mu_std": np.nan})
            if self.n_samples > 0:
                sims = np.asarray(sc.inverse_transform(m.sample(self.n_samples, burn=self.burn)))
                rec["model_market_mean"] = float(market_mean(sims).mean())
                rec["model_market_kurtosis"] = float(np.nanmean(market_kurtosis(sims)))
            rows.append(rec)
            if self.keep_models:
                self.models_.append((win.index[-1], m, sc))
            prev = m
        self.last_model_, self.last_scaler_ = prev, sc
        self.results_ = pd.DataFrame(rows).set_index("date")
        return self
