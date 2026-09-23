"""Volatility filters: model the dependence on standardised returns.

A single phi^4 site with lambda >= 0 has tails no heavier than a Gaussian, so
fat-tailed daily returns are better handled in two steps: divide each return
by a one-step-ahead volatility forecast sigma_{t|t-1}, fit phi^4 to the
standardised series z_t = r_t / sigma_{t|t-1}, and multiply back.

Every sigma here uses information up to t-1 only (no look-ahead), and
``forecast`` gives sigma_{T+1|T} for the day after the sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _as_2d(r):
    if isinstance(r, pd.Series):
        return r.to_frame(), True
    if isinstance(r, pd.DataFrame):
        return r, False
    a = np.asarray(r, float)
    return (pd.DataFrame(a[:, None]) if a.ndim == 1 else pd.DataFrame(a)), a.ndim == 1


class EWMAVol:
    """RiskMetrics volatility: sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}.

    The recursion starts from the variance of the first ``init`` returns.
    """

    def __init__(self, lam: float = 0.94, init: int = 30):
        if not 0 < lam < 1:
            raise ValueError("lam must be in (0, 1)")
        self.lam, self.init = float(lam), int(init)

    def fit(self, returns):
        return self          # nothing to estimate

    def _path(self, x):
        x = np.asarray(x, float)
        v = np.empty(len(x) + 1)
        v[0] = np.var(x[:self.init]) if len(x) > 1 else x[0] ** 2
        for t in range(len(x)):
            v[t + 1] = self.lam * v[t] + (1 - self.lam) * x[t] ** 2
        return np.sqrt(v)                     # v[t] = sigma^2_{t|t-1}; last = forecast

    def sigma(self, returns):
        """sigma_{t|t-1} aligned with ``returns`` (same shape and index)."""
        df, single = _as_2d(returns)
        out = df.apply(lambda c: pd.Series(self._path(c.to_numpy())[:-1], index=c.index))
        return out.iloc[:, 0] if single else out

    def forecast(self, returns):
        """sigma_{T+1|T} for each column."""
        df, single = _as_2d(returns)
        f = pd.Series({c: self._path(df[c].to_numpy())[-1] for c in df.columns})
        return float(f.iloc[0]) if single else f


class GARCHVol:
    """GARCH(1,1) with Gaussian quasi-likelihood, fitted per column:

        sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1},  alpha + beta < 1.
    """

    def __init__(self):
        self.params_ = {}

    @staticmethod
    def _var_path(x, omega, alpha, beta):
        v = np.empty(len(x) + 1)
        v[0] = np.var(x)
        for t in range(len(x)):
            v[t + 1] = omega + alpha * x[t] ** 2 + beta * v[t]
        return v

    def _fit_one(self, x):
        x = np.asarray(x, float) - np.mean(x)
        s2 = np.var(x)

        def nll(th):
            a, b = th
            if a < 0 or b < 0 or a + b >= 0.999:
                return 1e10
            v = self._var_path(x, s2 * (1 - a - b), a, b)[:-1]
            return 0.5 * np.sum(np.log(v) + x**2 / v)

        best = min((minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6})
                    for x0 in ([0.05, 0.90], [0.10, 0.80], [0.03, 0.95])), key=lambda r: r.fun)
        a, b = best.x
        return {"omega": s2 * (1 - a - b), "alpha": a, "beta": b, "mean": float(np.mean(x))}

    def fit(self, returns):
        df, _ = _as_2d(returns)
        self.params_ = {c: self._fit_one(df[c].to_numpy()) for c in df.columns}
        return self

    def _path(self, c, x):
        p = self.params_[c]
        return np.sqrt(self._var_path(np.asarray(x, float) - p["mean"], p["omega"], p["alpha"], p["beta"]))

    def sigma(self, returns):
        df, single = _as_2d(returns)
        out = pd.DataFrame({c: self._path(c, df[c].to_numpy())[:-1] for c in df.columns}, index=df.index)
        return out.iloc[:, 0] if single else out

    def forecast(self, returns):
        df, single = _as_2d(returns)
        f = pd.Series({c: self._path(c, df[c].to_numpy())[-1] for c in df.columns})
        return float(f.iloc[0]) if single else f


def make_vol(kind):
    """``"ewma"``, ``"garch"``, a filter instance, or None."""
    if kind is None or hasattr(kind, "sigma"):
        return kind
    return {"ewma": EWMAVol, "garch": GARCHVol}[kind]()


def devolatilize(returns, vol="ewma"):
    """Standardised returns z_t = r_t / sigma_{t|t-1} and the sigma used.

    Returns (z, sigma, fitted_filter). The first rows depend on the filter's
    start-up and are best dropped (e.g. the first 30).
    """
    f = make_vol(vol).fit(returns)
    sig = f.sigma(returns)
    return returns / sig, sig, f
