import warnings

import numpy as np
import pandas as pd
import pytest

import phi4finance.data as data_mod
from phi4finance import RollingPhi4, lag_embed, load_prices, load_returns
from phi4finance.baselines import (ols_fit_predict, rescaled_mean, ridge_ar_forecast,
                                   rolling_ar_forecast)
from phi4finance.metrics import coverage, hit_rate, mae, mae_se, sign_agreement, sign_product_matrix
from phi4finance.validation import select_l2


def _fake_download(calls):
    def f(tickers, start, end, period, interval, adjusted):
        calls.append(tuple(tickers))
        idx = pd.bdate_range("2020-01-01", periods=60)
        rng = np.random.default_rng(0)
        df = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, len(tickers))), 0)),
                          index=idx, columns=list(tickers))
        if "NEW" in df:
            df.loc[idx[:40], "NEW"] = np.nan     # listed late
        return df
    return f


def test_load_prices_cache_and_coverage(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(data_mod, "_download", _fake_download(calls))
    with pytest.warns(UserWarning, match="NEW"):
        p = load_prices(["A", "B", "NEW"], start="2020-01-01", end="2020-04-01", cache_dir=tmp_path)
    assert list(p.columns) == ["A", "B"] and len(p) == 60
    p2 = load_prices(["A", "B", "NEW"], start="2020-01-01", end="2020-04-01", cache_dir=tmp_path)
    assert len(calls) == 1                      # second call served from the CSV cache
    assert np.allclose(p2.values, p.values)
    r = load_returns(["A", "B"], start="2020-01-01", end="2020-04-01", cache_dir=tmp_path, log=False)
    assert r.shape == (59, 2)


def test_rolling_phi4_records_windows():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-01-01", periods=300)
    f = rng.standard_t(4, 300) * 0.01
    R = pd.DataFrame(0.7 * f[:, None] + rng.standard_t(4, (300, 4)) * 0.01, index=idx,
                     columns=list("ABCD"))
    roll = RollingPhi4(window=100, step=50, n_samples=400, burn=50, verbose=False,
                       model_kw={"n_chains": 8}).fit(R)
    res = roll.results_
    assert len(res) == 5 and res.index[-1] == idx[-1]
    assert np.isfinite(res[["model_market_kurtosis", "data_market_kurtosis"]].values).all()
    assert (res.w_mean > 0).all()          # common factor -> positive couplings
    assert roll.last_model_.V == 4


def test_baselines():
    # eq. 10 with two predictors
    r = rescaled_mean([[0.02, -0.01]], [0.02, 0.01], 0.03)
    assert np.isclose(r[0], 0.03 / 2 * (1 - 1))
    X = np.random.default_rng(0).normal(size=(200, 2))
    y = 0.5 + X @ [1.0, -2.0]
    assert np.allclose(ols_fit_predict(X, y, [[1.0, 1.0]]), 0.5 - 1.0)
    rng = np.random.default_rng(1)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.6 * x[t - 1] + rng.normal()
    pred = rolling_ar_forecast(x, [1500, 1800], window=1000)
    assert np.allclose(pred, 0.6 * x[[1499, 1799]], atol=0.15)
    E = lag_embed(x[:500], 6)
    assert ridge_ar_forecast(E, E[:3, 1:], alpha=1.0).shape == (3,)


def test_metrics_for_paper():
    m = sign_product_matrix([0.1, -0.2, 0.3])
    assert m[0, 1] == -1 and m[0, 2] == 1
    assert sign_agreement(m, m) == 1.0
    assert mae([1, 2], [1, 4]) == 1.0 and np.isfinite(mae_se([1, 2, 3], [0, 0, 0]))
    assert coverage([0, 5], [-1, -1], [1, 1]) == 0.5
    assert hit_rate([1, -1], [2, 3]) == 0.5


def test_select_l2_picks_from_grid():
    rng = np.random.default_rng(0)
    X = lag_embed(np.tanh(rng.normal(0, 0.2, 200)), 8)
    best, table = select_l2(X, target_idx=7, l2_grid=(0.01, 1.0), n_val=10)
    assert best in (0.01, 1.0) and len(table) == 2
    with pytest.raises(ValueError):
        select_l2(X, 7, n_val=0)
