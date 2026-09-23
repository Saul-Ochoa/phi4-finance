import warnings

import numpy as np
import pandas as pd
import pytest

from phi4finance import ConditionalDistribution, Phi4Model
from phi4finance.backtest import GaussianVolForecaster, Phi4LaggedForecaster, summarize, walk_forward
from phi4finance.estimators import neg_pseudo_loglik, _Packer
from phi4finance.inference import site_grid
from phi4finance.metrics import crps_gaussian, diebold_mariano
from phi4finance.structure import Tying, lag_embed_panel
from phi4finance.validation import select_l2
from phi4finance.volatility import EWMAVol, GARCHVol, devolatilize


# ------------------------------------------------------------------ tying
def test_tying_constructors():
    t = Tying.free(5)
    assert t.is_free and t.n_w == 10 and t.n_site == 5
    tt = Tying.toeplitz(6, max_lag=2)
    assert tt.n_w == 2 and tt.w_groups[0, 3] == -1 and tt.w_groups[0, 1] == tt.w_groups[3, 4]
    tl = Tying.lagged(2, 2)                         # 3 days x 2 assets
    assert tl.V == 6 and tl.n_site == 2
    # stationarity: (a, day0)->(b, day1) shares a group with (a, day1)->(b, day2)
    assert tl.w_groups[0, 3] == tl.w_groups[2, 5]
    # direction matters across days: a leads b != b leads a
    assert tl.w_groups[0, 3] != tl.w_groups[1, 2]
    # same-day pairs are symmetric and shared across days
    assert tl.w_groups[0, 1] == tl.w_groups[2, 3] == tl.w_groups[4, 5]
    tg = Tying.lagged(3, 2, target=1)
    assert tg.V == 7 and tg.site_groups[-1] == 1


def test_lag_embed_panel_matches_tying_order():
    Z = np.arange(20, dtype=float).reshape(10, 2)   # value = 2*day + asset
    X = lag_embed_panel(Z, 2, target=1)
    assert X.shape == (8, 5)
    assert X[0].tolist() == [0, 1, 2, 3, 5]         # day0 a0,a1, day1 a0,a1, day2 a1
    assert lag_embed_panel(Z, 2).shape == (8, 6)


def test_tied_pl_gradient_finite_differences():
    V_t = Tying.lagged(2, 2, target=0)
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.8, 0.8, (40, V_t.V))
    m = Phi4Model(V_t.V, tying=V_t, mu_global=False, lam_global=False, seed=1)
    m.W = V_t.expand_w(rng.normal(0, 0.2, V_t.n_w)); m.a = V_t.expand_site(rng.normal(0, 0.2, V_t.n_site))
    pk = _Packer(V_t, False, False, set())
    grid = site_grid(3.0, 201)
    x0 = pk.pack(m.params())
    scale = X.std(0)
    f0, g = neg_pseudo_loglik(pk.unpack(x0, m.params()), X, grid, l2=0.3, scale=scale)
    gflat = pk.flat_grad(g)
    for idx in [0, 3, V_t.n_w + 1, len(x0) - 1]:
        x1 = x0.copy(); x1[idx] += 1e-6
        f1, _ = neg_pseudo_loglik(pk.unpack(x1, m.params()), X, grid, l2=0.3, scale=scale)
        assert np.isclose((f1 - f0) / 1e-6, gflat[idx], rtol=1e-3, atol=1e-5)


def test_tied_model_keeps_structure_and_recovers_toeplitz():
    rng = np.random.default_rng(0)
    phi = 0.5
    x = np.zeros(6000)
    for t in range(1, 6000):
        x[t] = phi * x[t - 1] + rng.normal(0, 1)
    X = lag_embed_panel(np.tanh(x / 4)[:, None], 3)           # 4 consecutive days
    t4 = Tying.toeplitz(4)
    m = Phi4Model(4, tying=t4, mu_global=False, lam_global=False).fit(X, method="pl", verbose=False)
    w = t4.reduce_w(m.W, "mean")
    assert np.allclose(m.W, t4.expand_w(w))                    # tied after fit
    assert w[0] > 0.2 and abs(w[1]) < 0.3 * w[0]              # AR(1): only lag 1 matters (partial)
    m.fit(X, method="ml", epochs=3, mcmc_steps=10, verbose=False)
    assert np.allclose(m.W, t4.expand_w(t4.reduce_w(m.W, "mean")))


# ------------------------------------------------------------------ scale-free penalty
def test_penalty_is_scale_free():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(300, 1))
    X = np.tanh(0.25 * (f + rng.normal(size=(300, 4))))
    fits = []
    for c in (1.0, 2.0):
        m = Phi4Model(4, mu_global=False, lam_global=False).fit(c * X / 2, method="pl", l2=0.5, verbose=False)
        fits.append(m.W / np.sqrt(np.outer(m.mu, m.mu)))       # implied partial correlations
    assert np.allclose(fits[0], fits[1], atol=0.02)
    m_old = [Phi4Model(4, mu_global=False, lam_global=False).fit(c * X / 2, method="pl", l2=0.5,
                                                                  penalty_scale="none", verbose=False)
             for c in (1.0, 2.0)]
    old = [m.W / np.sqrt(np.outer(m.mu, m.mu)) for m in m_old]
    assert not np.allclose(old[0], old[1], atol=0.02)          # the v0.4 penalty was not


def test_negative_lambda_with_pl_warns():
    X = np.random.default_rng(0).uniform(-0.5, 0.5, (50, 3))
    with pytest.warns(UserWarning, match="lam_min < 0"):
        Phi4Model(3, lam_min=-5.0).fit(X, method="pl", verbose=False)


# ------------------------------------------------------------------ volatility
def _garch_series(n=3000, seed=0, om=1e-6, a=0.08, b=0.9):
    rng = np.random.default_rng(seed); v = om / (1 - a - b); r = np.zeros(n)
    for t in range(n):
        r[t] = np.sqrt(v) * rng.standard_normal(); v = om + a * r[t] ** 2 + b * v
    return r


def test_ewma_is_one_step_ahead():
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01])
    e = EWMAVol(lam=0.9, init=2)
    s = e.sigma(r)
    v0 = np.var(r[:2])
    assert np.isclose(s.iloc[0] ** 2, v0)
    assert np.isclose(s.iloc[1] ** 2, 0.9 * v0 + 0.1 * 0.01 ** 2)    # uses r_0 only
    assert np.isclose(e.forecast(r) ** 2, 0.9 * s.iloc[-1] ** 2 + 0.1 * 0.01 ** 2)


def test_garch_recovers_parameters_and_devol_reduces_kurtosis():
    from scipy.stats import kurtosis
    r = _garch_series()
    p = GARCHVol().fit(r).params_[0]
    assert abs(p["alpha"] - 0.08) < 0.04 and abs(p["beta"] - 0.9) < 0.05
    z, sig, _ = devolatilize(pd.Series(r), "garch")
    assert kurtosis(z[30:], fisher=False) < kurtosis(r, fisher=False)


# ------------------------------------------------------------------ scores
def test_crps_logscore_pit():
    d = ConditionalDistribution.gaussian(0.1, 0.5)
    for y in (-1.0, 0.1, 0.7, 5.0):
        assert np.isclose(d.crps(y), crps_gaussian(0.1, 0.5, y), atol=1e-4)
    assert np.isclose(d.logpdf(0.1), -np.log(0.5 * np.sqrt(2 * np.pi)), atol=1e-3)
    assert np.isclose(d.pit(0.1), 0.5, atol=1e-3)
    assert d.logpdf(50.0) == -np.inf


def test_diebold_mariano():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 500)
    assert diebold_mariano(a, a)[1] == 1.0
    stat, p = diebold_mariano(a - 0.05 + rng.normal(0, 0.1, 500), a, h=2)
    assert stat < 0 and p < 0.01


def test_select_l2_blocked_cv():
    rng = np.random.default_rng(0)
    X = np.tanh(rng.normal(0, 0.3, (120, 5)))
    best, table = select_l2(X, 4, l2_grid=(0.01, 10.0), folds=3, metric="crps")
    assert best in (0.01, 10.0) and "val_crps" in table


# ------------------------------------------------------------------ backtest
def test_walk_forward_and_summary():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=330)
    r = pd.DataFrame({"A": _garch_series(330, 1), "B": _garch_series(330, 2)}, index=idx)
    fc = {"gauss": GaussianVolForecaster("ewma"),
          "phi4": Phi4LaggedForecaster(n_lags=2, l2=1.0),
          "phi4_cv": Phi4LaggedForecaster(n_lags=2, l2_grid=(0.1, 10.0), folds=2, cross=False)}
    res = walk_forward(r, fc, start=idx[300], refit_every=15, train_window=250, verbose=False)
    assert len(res) == 30 * 2 * 3
    assert np.isfinite(res.crps).all() and res.pit.between(0, 1).all()
    s = summarize(res, "gauss")
    assert s.loc["gauss", "CRPS vs bench"] == 1.0 and 0 <= s.loc["phi4", "coverage"] <= 1
    assert len(summarize(res, "gauss", by_asset=True)) == 6
