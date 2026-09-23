"""Walk-forward evaluation of next-day predictive distributions.

A *forecaster* has ``fit(train)`` (train: DataFrame of returns, rows = days,
columns = assets) and ``predict(history) -> {asset: ConditionalDistribution}``
for the day after ``history``. ``walk_forward`` refits every ``refit_every``
days on the preceding ``train_window`` days and scores every forecast;
``summarize`` compares methods with MAE, CRPS, interval coverage and
Diebold-Mariano tests against a benchmark.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .inference import ConditionalDistribution
from .metrics import diebold_mariano
from .model import Phi4Model
from .structure import Tying, lag_embed_panel
from .validation import DEFAULT_L2_GRID, select_l2
from .volatility import make_vol


class GaussianVolForecaster:
    """N(m, sigma^2_{t+1|t}): the distributional version of the zero forecast.
    ``mean="zero"`` (default) or ``"train"`` (training-window mean);
    ``vol="ewma"`` or ``"garch"``."""

    def __init__(self, vol="ewma", mean: str = "zero"):
        self.vol_kind, self.mean = vol, mean

    def fit(self, train: pd.DataFrame):
        self.vol_ = make_vol(self.vol_kind).fit(train)
        self.mu_ = train.mean() if self.mean == "train" else pd.Series(0.0, index=train.columns)
        return self

    def predict(self, history: pd.DataFrame) -> dict:
        sig = self.vol_.forecast(history)
        return {c: ConditionalDistribution.gaussian(float(self.mu_[c]), float(sig[c])) for c in history.columns}


class Phi4LaggedForecaster:
    """phi^4 on volatility-standardised returns over ``n_lags`` past days.

    For every target asset k one model is fitted whose sites are the past
    ``n_lags`` days of the inputs plus asset k on the forecast day, with
    block-Toeplitz tying (``Tying.lagged``). The forecast is the exact 1-D
    conditional, mapped back to returns with sigma_{t+1|t}.

    cross : True uses every asset's lags as inputs; False only the target's own
        lags (a univariate Toeplitz model on standardised returns).
    z_clip : standardised returns are clipped to +-z_clip and divided by it, so
        the model works on [-1, 1] inside the [-1.5, 1.5] support.
    l2 : fixed penalty, or None to choose it by blocked CV on CRPS every
        ``reselect_every`` fits.
    zero_mean : keep the biases a_i at 0 (default). A window's average return
        is mostly noise at the daily horizon, and estimating it biases the
        forecast mean.
    """

    def __init__(self, n_lags: int = 5, vol="ewma", cross: bool = True, l2=None,
                 l2_grid=DEFAULT_L2_GRID, folds: int = 4, reselect_every: int = 3,
                 z_clip: float = 5.0, vol_burn: int = 30, max_lag=None, n_grid: int = 101,
                 zero_mean: bool = True):
        self.n_lags, self.vol_kind, self.cross = int(n_lags), vol, cross
        self.l2, self.l2_grid, self.folds, self.reselect_every = l2, tuple(l2_grid), folds, reselect_every
        self.z_clip, self.vol_burn, self.max_lag, self.n_grid = z_clip, vol_burn, max_lag, n_grid
        self.zero_mean = zero_mean
        self._n_fits, self.l2_ = 0, {}

    def _z(self, R):
        sig = self.vol_.sigma(R)
        return (R / sig).clip(-self.z_clip, self.z_clip) / self.z_clip

    def fit(self, train: pd.DataFrame):
        self.vol_ = make_vol(self.vol_kind).fit(train)
        Z = self._z(train).iloc[self.vol_burn:]
        self.assets_ = list(train.columns)
        self.models_ = {}
        reselect = self.l2 is None and self._n_fits % self.reselect_every == 0
        for k, a in enumerate(self.assets_):
            inputs = Z if self.cross else Z[[a]]
            tk = k if self.cross else 0
            X = lag_embed_panel(inputs.to_numpy(), self.n_lags, target=tk)
            tying = Tying.lagged(inputs.shape[1], self.n_lags, target=tk, max_lag=self.max_lag)
            mkw = {"tying": tying, "mu_global": False, "lam_global": False,
                   "freeze": ("a",) if self.zero_mean else ()}
            if self.l2 is not None:
                l2 = self.l2
            elif reselect or a not in self.l2_:
                l2, _ = select_l2(X, X.shape[1] - 1, l2_grid=self.l2_grid, folds=self.folds, metric="crps",
                                  model_kw=mkw, fit_kw={"n_grid": self.n_grid})
            else:
                l2 = self.l2_[a]
            self.l2_[a] = l2
            self.models_[a] = Phi4Model(X.shape[1], **mkw).fit(X, method="pl", l2=l2, n_grid=self.n_grid,
                                                               verbose=False)
        self._n_fits += 1
        return self

    def predict(self, history: pd.DataFrame) -> dict:
        hist = history[self.assets_]
        sig_next = self.vol_.forecast(hist)
        Zh = self._z(hist).iloc[-self.n_lags:]
        out = {}
        for k, a in enumerate(self.assets_):
            past = (Zh if self.cross else Zh[[a]]).to_numpy().ravel()      # oldest day first
            m = self.models_[a]
            d = m.conditional_distribution({i: v for i, v in enumerate(past)}, m.V - 1)
            out[a] = d.affine(self.z_clip * float(sig_next[a]))
        return out


def walk_forward(returns: pd.DataFrame, forecasters: dict, start, end=None, refit_every: int = 10,
                 train_window: int = 250, level: float = 0.9, verbose: bool = True) -> pd.DataFrame:
    """Score next-day forecasts for every day in [start, end].

    Every ``refit_every`` days each forecaster is refitted on the
    ``train_window`` days before; each forecast uses all returns up to the
    previous day. Returns one row per (date, asset, method) with the actual
    return, mean, central interval, CRPS, log score and PIT.
    """
    R = returns.sort_index()
    end_ts = pd.Timestamp(end) if end is not None else R.index[-1]
    days = R.index[(R.index >= pd.Timestamp(start)) & (R.index <= end_ts)]
    if len(days) == 0:
        raise ValueError("no evaluation days in range")
    rows, t0 = [], time.time()
    for b in range(0, len(days), refit_every):
        block = days[b:b + refit_every]
        pos = R.index.get_loc(block[0])
        if pos < train_window:
            raise ValueError(f"need {train_window} days before {block[0].date()}")
        train = R.iloc[pos - train_window:pos]
        for f in forecasters.values():
            f.fit(train)
        for day in block:
            hist = R.loc[:day].iloc[:-1]
            y = R.loc[day]
            for name, f in forecasters.items():
                for asset, dist in f.predict(hist).items():
                    lo, hi = dist.interval(level)
                    yv = float(y[asset])
                    rows.append({"date": day, "asset": asset, "method": name, "actual": yv,
                                 "mean": dist.mean(), "std": dist.std(), "lo": lo, "hi": hi,
                                 "crps": dist.crps(yv), "logscore": float(dist.logpdf(yv)),
                                 "pit": float(dist.pit(yv))})
        if verbose:
            print(f"  {block[-1].date()}: {min(b + refit_every, len(days))}/{len(days)} days, {time.time() - t0:.0f} s")
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, benchmark: str, by_asset: bool = False) -> pd.DataFrame:
    """MAE, CRPS, coverage and Diebold-Mariano p-values against ``benchmark``
    (negative DM statistic = better than the benchmark)."""
    df = results.assign(abs_err=(results["mean"] - results["actual"]).abs(),
                        covered=(results.actual >= results.lo) & (results.actual <= results.hi))
    keys = ["asset", "method"] if by_asset else ["method"]
    bench = df[df.method == benchmark].set_index(["date", "asset"])
    out = []
    for key, g in df.groupby(keys):
        g2 = g.set_index(["date", "asset"])
        b = bench.loc[g2.index]
        dm_crps = diebold_mariano(g2.crps.to_numpy(), b.crps.to_numpy())
        dm_mae = diebold_mariano(g2.abs_err.to_numpy(), b.abs_err.to_numpy())
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec.update({"n": len(g), "MAE": g.abs_err.mean(), "CRPS": g.crps.mean(),
                    "CRPS vs bench": g.crps.mean() / b.crps.mean(), "p DM (CRPS)": dm_crps[1],
                    "p DM (MAE)": dm_mae[1], "coverage": g.covered.mean(),
                    "mean log score": g.logscore.replace(-np.inf, np.nan).mean(),
                    "PIT std (0.289 if calibrated)": g.pit.std()})
        out.append(rec)
    return pd.DataFrame(out).set_index(keys)
