import numpy as np
import pandas as pd

from phi4finance import RollingPhi4
from phi4finance.earlywarning import (absorption_ratio, auc, average_correlation, forward_max_drawdown,
                                      forward_realized_vol, hac_ols, oos_r2, rolling_indicator)


def test_indicators():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(2000, 1))
    X = f + rng.normal(size=(2000, 10))                  # one factor, corr 0.5
    assert abs(average_correlation(X) - 0.5) < 0.03
    assert absorption_ratio(X, k=1) > 0.5 > absorption_ratio(rng.normal(size=(2000, 10)), k=1)
    R = pd.DataFrame(X, index=pd.bdate_range("2020-01-01", periods=2000))
    s = rolling_indicator(R, average_correlation, 100, R.index[[50, 500]])
    assert np.isnan(s.iloc[0]) and abs(s.iloc[1] - 0.5) < 0.2


def test_forward_targets_use_only_future():
    r = pd.Series([0.0, 0.1, -0.1, 0.0, 0.0, 0.0])
    fv = forward_realized_vol(r, h=2, annualize=1)
    assert np.isclose(fv.iloc[0], r.iloc[1:3].std())                 # t+1..t+2
    dd = forward_max_drawdown(r, h=2)
    assert np.isclose(dd.iloc[0], 0.1) and np.isnan(dd.iloc[-1])      # 1.1 -> 0.99


def test_hac_ols_and_oos_r2():
    rng = np.random.default_rng(1)
    n = 600
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 1 + 0.5 * x1 + 0.3 * x2 + rng.normal(size=n)
    res = hac_ols(y, pd.DataFrame({"x1": x1, "x2": x2}), lags=5)
    assert abs(res.loc["x1", "coef"] - 0.5) < 0.1 and res.loc["x2", "p"] < 0.01
    good = oos_r2(y, x1[:, None], np.column_stack([x1, x2]), min_train=200, gap=5)
    bad = oos_r2(y, x1[:, None], np.column_stack([x1, rng.normal(size=n)]), min_train=200, gap=5)
    assert good["r2_oos"] > 0.03 and good["p_cw"] < 0.01
    assert bad["r2_oos"] < good["r2_oos"]


def test_auc():
    assert auc([3, 4, 1, 2], [True, True, False, False]) == 1.0
    assert auc([1, 2, 3, 4], [True, True, False, False]) == 0.0


def test_rolling_records_scale_free_couplings():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(300, 1))
    R = pd.DataFrame(0.01 * (f + rng.normal(size=(300, 4))), index=pd.bdate_range("2021-01-01", periods=300))
    res = RollingPhi4(window=120, step=60, n_samples=0, verbose=False,
                      model_kw={"mu_global": False, "lam_global": False}).fit(R).results_
    assert {"coupling_mean", "hub_strength_max", "log_mu_std"} <= set(res.columns)
    assert (res.coupling_mean > 0.1).all()
