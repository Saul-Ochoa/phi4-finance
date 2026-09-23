"""Early-warning evaluation: do risk indicators anticipate turbulence?

Indicators computed on a trailing window at date t are compared with what
happens over the next ``h`` days (realised volatility, maximum drawdown). The
key baseline is the *current* volatility: volatility clusters, so an
indicator is only useful if it adds information beyond it.

Benchmarks from the literature:
* average pairwise correlation;
* absorption ratio (Kritzman, Li, Page & Rigobon, 2011): share of total
  variance explained by the first ``k`` eigenvectors of the covariance matrix
  (k ~ one fifth of the number of assets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------- indicators
def average_correlation(R) -> float:
    C = np.corrcoef(np.asarray(R, float).T)
    iu = np.triu_indices(C.shape[0], 1)
    return float(C[iu].mean())


def absorption_ratio(R, k=None) -> float:
    """Fraction of total variance absorbed by the top ``k`` eigenvectors."""
    X = np.asarray(R, float)
    k = k or max(1, X.shape[1] // 5)
    ev = np.linalg.eigvalsh(np.cov(X.T))[::-1]
    return float(ev[:k].sum() / ev.sum())


def rolling_indicator(R: pd.DataFrame, fn, window: int, dates) -> pd.Series:
    """``fn`` applied to the ``window`` rows ending at each date in ``dates``."""
    pos = R.index.get_indexer(dates)
    return pd.Series([fn(R.iloc[p - window + 1:p + 1]) if p >= window - 1 else np.nan for p in pos], index=dates)


# ---------------------------------------------------------------- targets
def forward_realized_vol(r: pd.Series, h: int = 20, annualize: int = 252) -> pd.Series:
    """Volatility of r over t+1 .. t+h (NaN where fewer than h days remain)."""
    fwd = r[::-1].rolling(h).std()[::-1].shift(-1)
    return fwd * np.sqrt(annualize)


def forward_max_drawdown(r: pd.Series, h: int = 20) -> pd.Series:
    """Largest peak-to-trough loss of the cumulative return over t+1 .. t+h, as a positive number."""
    x = r.to_numpy()
    out = np.full(len(x), np.nan)
    for t in range(len(x) - h):
        w = np.cumprod(1 + x[t + 1:t + 1 + h])
        w = np.concatenate([[1.0], w])
        out[t] = float(-(w / np.maximum.accumulate(w) - 1).min())
    return pd.Series(out, index=r.index)


def trailing_realized_vol(r: pd.Series, window: int = 20, annualize: int = 252) -> pd.Series:
    return r.rolling(window).std() * np.sqrt(annualize)


# ---------------------------------------------------------------- statistics
def hac_ols(y, X, lags: int) -> pd.DataFrame:
    """OLS with an intercept and Newey-West (Bartlett) standard errors.

    X : DataFrame of regressors. Returns coef, se, t, p per regressor.
    """
    X = pd.DataFrame(X).copy()
    X.insert(0, "const", 1.0)
    Xv, yv = X.to_numpy(float), np.asarray(y, float)
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    b = XtX_inv @ Xv.T @ yv
    u = yv - Xv @ b
    Xu = Xv * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        G = Xu[L:].T @ Xu[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    t = b / se
    return pd.DataFrame({"coef": b, "se": se, "t": t, "p": 2 * (1 - norm.cdf(np.abs(t)))}, index=X.columns)


def oos_r2(y: pd.Series, X_base: pd.DataFrame, X_full: pd.DataFrame, min_train: int, gap: int) -> dict:
    """Out-of-sample R^2 of the full model relative to the base model.

    For each row t >= min_train both regressions are fitted on rows
    0 .. t - gap (``gap`` rows kept out so that overlapping targets never
    leak into training) and used to predict y_t. Returns
    R2_oos = 1 - SSE_full / SSE_base (> 0: the extra regressors help) and
    the Clark-West-style one-sided p-value of that improvement.
    """
    y = np.asarray(y, float)
    if not 0 < min_train < len(y) or min_train - gap < 5:
        raise ValueError(f"need 5 + gap <= min_train < len(y) (min_train={min_train}, gap={gap}, n={len(y)})")
    Xb = np.column_stack([np.ones(len(y)), np.asarray(X_base, float)])
    Xf = np.column_stack([np.ones(len(y)), np.asarray(X_full, float)])
    eb, ef, fb, ff = [], [], [], []
    for t in range(min_train, len(y)):
        tr = slice(0, t - gap + 1)
        bb = np.linalg.lstsq(Xb[tr], y[tr], rcond=None)[0]
        bf = np.linalg.lstsq(Xf[tr], y[tr], rcond=None)[0]
        pb, pf = Xb[t] @ bb, Xf[t] @ bf
        eb.append(y[t] - pb); ef.append(y[t] - pf); fb.append(pb); ff.append(pf)
    eb, ef, fb, ff = map(np.array, (eb, ef, fb, ff))
    r2 = 1 - (ef ** 2).sum() / (eb ** 2).sum()
    # Clark-West adjusted loss difference for nested models
    f = eb ** 2 - (ef ** 2 - (fb - ff) ** 2)
    se = np.sqrt(hac_var(f, lags=gap) / len(f))
    p = float(1 - norm.cdf(f.mean() / se)) if se > 0 else float("nan")
    return {"r2_oos": float(r2), "p_cw": p, "n": len(f)}


def hac_var(x, lags: int) -> float:
    x = np.asarray(x, float) - np.mean(x)
    v = x @ x / len(x)
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (x[L:] @ x[:-L]) / len(x)
    return float(v)


def auc(score, event) -> float:
    """Area under the ROC curve (probability a random event date scores higher than a random calm date)."""
    s, e = np.asarray(score, float), np.asarray(event, bool)
    pos, neg = s[e], s[~e]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
