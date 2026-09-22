"""Baselines for the prediction tasks of Sections 3.4 and 3.5."""
from __future__ import annotations

import numpy as np

from .preprocessing import lag_embed


def rescaled_mean(predictor_returns, predictor_std, target_std) -> np.ndarray:
    """Baseline R of eq. 10, generalised to k predictors:

        R_target = sigma_target / k * sum_j r_j / sigma_j

    predictor_returns : (n, k) same-day returns of the predictor stocks.
    predictor_std : (k,) standard deviations on the training window.
    target_std : standard deviation of the target on the training window.
    """
    r = np.atleast_2d(np.asarray(predictor_returns, dtype=float))
    s = np.asarray(predictor_std, dtype=float)
    return float(target_std) * (r / s).mean(axis=1)


def ols_fit_predict(X_train, y_train, X_test, alpha: float = 0.0) -> np.ndarray:
    """Linear regression with intercept (ridge on the slopes if ``alpha`` > 0)."""
    X = np.asarray(X_train, float); y = np.asarray(y_train, float)
    Xt = np.atleast_2d(np.asarray(X_test, float))
    if X.ndim == 1:
        X = X[:, None]
    if Xt.shape[1] != X.shape[1]:
        Xt = Xt.reshape(-1, X.shape[1])
    mx, my = X.mean(0), y.mean()
    Xc = X - mx
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - my))
    return my + (Xt - mx) @ beta


def rolling_ar_forecast(series, targets, window: int, lags: int = 1) -> np.ndarray:
    """One-step forecasts by OLS of r_s on (r_{s-1}, ..., r_{s-lags}) fitted on
    the ``window`` observations before each target index (all data up to the
    previous day, as the linear-regression baseline of Section 3.5)."""
    x = np.asarray(series, dtype=float)
    out = []
    for t in targets:
        lo = t - window - lags
        if lo < 0:
            raise ValueError(f"not enough history for target {t} with window={window}")
        E = lag_embed(x[lo:t], lags + 1)           # rows: lags past values, then the response
        out.append(ols_fit_predict(E[:, :-1], E[:, -1], x[t - lags:t])[0])
    return np.array(out)


def ridge_ar_forecast(train_rows, histories, alpha: float = 1.0) -> np.ndarray:
    """Ridge regression of the last column of ``train_rows`` (lag-embedded,
    as used for the phi^4 forecaster) on the other columns, then applied to
    each history (n, V-1). Same information set as the phi^4 model."""
    R = np.asarray(train_rows, float)
    return ols_fit_predict(R[:, :-1], R[:, -1], np.asarray(histories, float), alpha=alpha)
