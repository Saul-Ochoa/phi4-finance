import numpy as np
import pytest

from phi4finance import ConditionalDistribution, HeatBathSampler, MetropolisSampler, Phi4Model, Scaler
from phi4finance.estimators import neg_pseudo_loglik
from phi4finance.inference import exact_conditional, site_grid
from conftest import GRID


def _random_params(V, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.2, (V, V)); W = (W + W.T) / 2; np.fill_diagonal(W, 0)
    return {"W": W, "a": rng.normal(0, 0.3, V), "mu": rng.uniform(0.2, 1.0, V),
            "lam": rng.uniform(0.3, 1.0, V)}


# ------------------------------------------------------------------ samplers
@pytest.mark.parametrize("cls", [MetropolisSampler, HeatBathSampler])
def test_multichain_single_site_matches_exact(cls):
    mu, lam, a = -0.8, 0.6, 0.4
    s = cls(np.zeros((1, 1)), [mu], [lam], [a], rng=np.random.default_rng(0), n_chains=64)
    x = s.sample(n_samples=40000, burn=50)[:, 0]
    p = np.exp(a * GRID - mu * GRID**2 - lam * GRID**4); p /= p.sum()
    m = (p * GRID).sum()
    assert abs(x.mean() - m) < 0.03
    assert abs(x.var() - (p * (GRID - m) ** 2).sum()) < 0.03
    assert s.last_state.shape == (64, 1)


def test_metropolis_and_heatbath_agree_on_correlations():
    p = _random_params(6, seed=2)
    kw = dict(rng=np.random.default_rng(1), n_chains=128)
    xm = MetropolisSampler(p["W"], p["mu"], p["lam"], p["a"], **kw).sample(20000, burn=100)
    xh = HeatBathSampler(p["W"], p["mu"], p["lam"], p["a"], **kw).sample(20000, burn=50)
    assert np.abs(xm.T @ xm / len(xm) - xh.T @ xh / len(xh)).max() < 0.03


def test_batched_action_and_init_shapes():
    p = _random_params(4)
    s = MetropolisSampler(p["W"], p["mu"], p["lam"], p["a"], n_chains=3, rng=np.random.default_rng(0))
    phi = np.random.default_rng(1).uniform(-1, 1, (3, 4))
    assert np.allclose(s.action(phi), [s.action(r) for r in phi])
    assert s.sample(10, burn=1, init=np.zeros(4)).shape == (10, 4)
    assert s.sample(10, burn=1, init=phi).shape == (10, 4)
    with pytest.raises(ValueError):
        s.sample(10, init=np.zeros((2, 4)))


# ------------------------------------------------------------------ exact conditional
def test_exact_conditional_matches_mcmc():
    p = _random_params(5, seed=3)
    phi = np.array([0.3, -0.2, 0.1, 0.5, 0.0])
    d = exact_conditional(p["W"], p["mu"], p["lam"], p["a"], phi, 4)
    s = HeatBathSampler(p["W"], p["mu"], p["lam"], p["a"], n_chains=256, rng=np.random.default_rng(0))
    x = s.sample_conditional({i: phi[i] for i in range(4)}, n_samples=40000, burn=5)[:, 4]
    assert abs(d.mean() - x.mean()) < 0.01 and abs(d.std() - x.std()) < 0.01


def test_conditional_distribution_api():
    g = site_grid(3.0, 601)
    d = ConditionalDistribution(g, np.exp(-g**2 / (2 * 0.3**2)))   # ~N(0, 0.3^2)
    assert abs(d.mean()) < 1e-9 and abs(d.std() - 0.3) < 1e-3
    lo, hi = d.interval(0.9)
    assert np.isclose(hi, 1.645 * 0.3, atol=5e-3) and np.isclose(lo, -hi)
    assert np.isclose(d.cdf(0.0), 0.5, atol=1e-3)
    assert abs(d.sample(20000, rng=np.random.default_rng(0)).std() - 0.3) < 0.01
    e = d.affine(0.02, 0.001)
    assert np.isclose(e.mean(), 0.001) and np.isclose(e.std(), 0.006, rtol=1e-3)
    n = d.affine(-2.0)                       # decreasing grid
    assert np.isclose(n.quantile(0.95), 2 * hi, atol=1e-2)


def test_model_exact_path_and_units():
    m = Phi4Model(4, seed=0)
    m.W = _random_params(4)["W"]
    rets = np.random.default_rng(0).normal(0, 0.02, (100, 4))
    sc = Scaler("minmax").fit(rets)
    known = {0: 0.01, 1: -0.02, 2: 0.005}
    d = m.conditional_distribution(known, 3, scaler=sc)
    d_model = m.conditional_distribution({k: sc.transform(v, cols=k) for k, v in known.items()}, 3)
    assert np.isclose(d.mean(), sc.inverse_transform(d_model.mean(), cols=3))
    x = m.predict_conditional(known, 3, n_samples=20000, scaler=sc)          # auto -> exact
    assert abs(x.mean() - d.mean()) < 3 * d.std() / np.sqrt(20000) + 1e-9
    with pytest.raises(ValueError):
        m.conditional_distribution({0: 0.1}, 3)
    with pytest.raises(ValueError):
        m.predict_conditional({0: 0.1}, 3, method="exact")
    f = Phi4Model(6, seed=0).forecast_distribution(np.zeros(5))
    assert isinstance(f, ConditionalDistribution)


# ------------------------------------------------------------------ pseudo-likelihood
def test_pl_gradient_matches_finite_differences():
    V = 4
    p = _random_params(V, seed=5)
    X = np.random.default_rng(0).uniform(-1, 1, (30, V))
    grid = site_grid(3.0, 201)
    f0, g = neg_pseudo_loglik(p, X, grid, l2=0.1)
    eps = 1e-6
    for name, idx in [("W", (0, 2)), ("a", 1), ("mu", 3), ("lam", 0)]:
        q = {k: v.copy() for k, v in p.items()}
        if name == "W":
            q["W"][idx] += eps; q["W"][idx[::-1]] += eps
        else:
            q[name][idx] += eps
        num = (neg_pseudo_loglik(q, X, grid, l2=0.1)[0] - f0) / eps
        assert np.isclose(num, g[name][idx], rtol=1e-3, atol=1e-5), name


def test_pl_objective_is_a_proper_log_density():
    # one free site, W = 0: -PL/N equals the exact negative log-likelihood per row
    p = {"W": np.zeros((1, 1)), "a": np.array([0.2]), "mu": np.array([0.5]), "lam": np.array([0.7])}
    X = np.array([[0.1], [-0.4]])
    grid = site_grid(3.0, 2001)
    f, _ = neg_pseudo_loglik(p, X, grid)
    logZ = np.log(np.trapezoid(np.exp(0.2 * grid - 0.5 * grid**2 - 0.7 * grid**4), grid))
    exact = -np.mean(0.2 * X[:, 0] - 0.5 * X[:, 0] ** 2 - 0.7 * X[:, 0] ** 4 - logZ)
    assert np.isclose(f, exact, atol=1e-3)


def test_pl_recovers_known_couplings(truth):
    m = Phi4Model(truth["V"], mu_global=False, lam_global=False, seed=0)
    m.fit(truth["data"], method="pl", verbose=False)
    iu = np.triu_indices(truth["V"], 1)
    assert np.corrcoef(m.W[iu], truth["W"][iu])[0, 1] > 0.95
    assert np.abs(m.a - truth["a"]).mean() < 0.1
    assert np.all(np.abs(m.mu - truth["mu"]) < 0.3)
    assert np.all(np.abs(m.lam - truth["lam"]) < 0.3)
    assert m.history[-1]["method"] == "pl"


def test_pl_respects_global_freeze_l2_and_bounds():
    X = np.random.default_rng(0).uniform(-1, 1, (80, 5))
    m = Phi4Model(5, mu_global=True, lam_global=True, freeze=("lam",), lam_min=0.05, seed=0)
    lam0 = m.lam.copy()
    m.fit(X, method="pl", verbose=False)
    assert np.all(m.mu == m.mu[0]) and np.array_equal(m.lam, lam0)
    m2 = Phi4Model(5, seed=0).fit(X, method="pl", l2=10.0, verbose=False)
    m3 = Phi4Model(5, seed=0).fit(X, method="pl", l2=0.0, verbose=False)
    assert np.abs(m2.W).sum() < np.abs(m3.W).sum()
    assert np.all(m3.lam >= m3.lam_min)
    with pytest.raises(ValueError):
        Phi4Model(2).fit(np.full((5, 2), 2.0), method="pl", verbose=False)


def test_pl_then_ml_refines():
    X = np.random.default_rng(0).uniform(-1, 1, (80, 3))
    m = Phi4Model(3, seed=0).fit(X, method="pl", verbose=False)
    m.fit(X, method="ml", epochs=3, mcmc_steps=10, verbose=False)
    assert [h["method"] for h in m.history][-1] == "ml"
