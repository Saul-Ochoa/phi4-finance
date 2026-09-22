import numpy as np
import pytest

from phi4finance import MetropolisSampler
from conftest import GRID


def _random_model(V, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.2, (V, V)); W = (W + W.T) / 2; np.fill_diagonal(W, 0)
    return W, rng.uniform(0.3, 1.0, V), rng.uniform(0.3, 1.0, V), rng.normal(0, 0.3, V)


def test_delta_action_matches_full_action():
    W, mu, lam, a = _random_model(7)
    s = MetropolisSampler(W, mu, lam, a, rng=np.random.default_rng(0))
    phi = np.random.default_rng(1).uniform(-1, 1, 7)
    for i in range(7):
        new = 0.37
        prop = phi.copy(); prop[i] = new
        assert np.isclose(s.delta_action(phi, i, new), s.action(prop) - s.action(phi))


def test_single_site_matches_exact_density():
    mu, lam, a = -0.8, 0.6, 0.4  # double well, tilted
    s = MetropolisSampler(np.zeros((1, 1)), [mu], [lam], [a], rng=np.random.default_rng(0))
    x = s.sample(n_samples=40000, burn=500)[:, 0]
    p = np.exp(a * GRID - mu * GRID**2 - lam * GRID**4); p /= p.sum()
    m = (p * GRID).sum(); v = (p * (GRID - m) ** 2).sum()
    assert abs(x.mean() - m) < 0.03
    assert abs(x.var() - v) < 0.03


def test_n_samples_and_shape_honoured():
    W, mu, lam, a = _random_model(4)
    s = MetropolisSampler(W, mu, lam, a, rng=np.random.default_rng(0))
    assert s.sample(n_samples=123, burn=10, thin=2).shape == (123, 4)
    assert 0.0 < s.acceptance_rate <= 1.0


def test_conditional_updates_every_free_site():
    V = 10
    W, mu, lam, a = _random_model(V)
    s = MetropolisSampler(W, mu, lam, a, rng=np.random.default_rng(0))
    out = s.sample_conditional({0: 0.2, 1: -0.1}, n_samples=500, burn=50)
    assert np.all(out[:, 0] == 0.2) and np.all(out[:, 1] == -0.1)
    # v0.1 left non-target free sites frozen at their initial values
    assert np.all(out[:, 2:].std(axis=0) > 0.05)


def test_conditional_matches_exact_when_one_site_free():
    V = 5
    W, mu, lam, a = _random_model(V, seed=3)
    known = {0: 0.3, 1: -0.2, 2: 0.1, 3: 0.5}
    phi = np.array([0.3, -0.2, 0.1, 0.5, 0.0])
    h = a[4] + 2 * W[4] @ phi
    p = np.exp(h * GRID - mu[4] * GRID**2 - lam[4] * GRID**4); p /= p.sum()
    s = MetropolisSampler(W, mu, lam, a, rng=np.random.default_rng(0))
    x = s.sample_conditional(known, n_samples=30000, burn=200)[:, 4]
    assert abs(x.mean() - (p * GRID).sum()) < 0.02


def test_input_validation():
    W, mu, lam, a = _random_model(3)
    with pytest.raises(ValueError):
        MetropolisSampler(W + np.triu(np.ones((3, 3)), 1), mu, lam, a)
    s = MetropolisSampler(W, mu, lam, a)
    with pytest.raises(ValueError):
        s.sample(fixed={0: 0, 1: 0, 2: 0})
