import numpy as np
import pytest

from phi4finance import Phi4Model, Scaler


class _FixedSampler:
    """Stands in for the MCMC so one update's direction can be checked exactly."""
    def __init__(self, samples):
        self.samples = samples
        self.acceptance_rate = 1.0
        self.last_state = samples[-1]

    def sample(self, **kw):
        return self.samples


def test_update_direction_is_gradient_ascent(monkeypatch):
    rng = np.random.default_rng(0)
    data = rng.normal(0.2, 0.2, (200, 3))          # narrow, positive mean
    model_samples = rng.normal(0.0, 0.6, (200, 3))  # wide, zero mean
    m = Phi4Model(3, lr=0.1, mu_global=False, lam_global=False, seed=0)
    monkeypatch.setattr(m, "sampler", lambda: _FixedSampler(model_samples))
    mu0, lam0, a0 = m.mu.copy(), m.lam.copy(), m.a.copy()
    m.fit(data, epochs=1, mcmc_steps=10, verbose=False)
    assert np.all(m.mu > mu0)    # model too wide -> stronger mass term
    assert np.all(m.lam > lam0)  # model tails too heavy -> stronger quartic term
    assert np.all(m.a > a0)      # data mean above model mean -> positive bias


def test_recovers_known_couplings(truth):
    m = Phi4Model(truth["V"], lr=0.1, mu_global=False, lam_global=False, seed=0)
    m.fit(truth["data"], epochs=250, mcmc_steps=200, batch_size=256, verbose=False)
    iu = np.triu_indices(truth["V"], 1)
    assert np.corrcoef(m.W[iu], truth["W"][iu])[0, 1] > 0.9
    assert np.abs(m.a - truth["a"]).mean() < 0.15
    assert np.all(np.abs(m.mu - truth["mu"]) < 0.4)
    assert np.all(np.abs(m.lam - truth["lam"]) < 0.4)
    h = m.history[-1]
    assert abs(h["m_model"] - h["m_data"]) < 0.05


def test_freeze_and_lambda_floor():
    data = np.random.default_rng(0).uniform(-1, 1, (50, 4))
    m = Phi4Model(4, lr=0.5, freeze=("mu",), lam_min=0.2, seed=0)
    mu0 = m.mu.copy()
    m.fit(data, epochs=5, mcmc_steps=30, verbose=False)
    assert np.array_equal(m.mu, mu0)
    assert np.all(m.lam >= 0.2)
    with pytest.raises(ValueError):
        Phi4Model(4, freeze=("beta",))


def test_small_dataset_full_batch_and_history():
    data = np.random.default_rng(0).uniform(-1, 1, (20, 3))
    m = Phi4Model(3, seed=0).fit(data, epochs=3, mcmc_steps=20, batch_size=64, verbose=False)
    assert len(m.history) == 3 and {"m_data", "chi_model"} <= set(m.history[0])


def test_seed_reproducible():
    data = np.random.default_rng(0).uniform(-1, 1, (30, 3))
    a = Phi4Model(3, seed=7).fit(data, epochs=3, mcmc_steps=20, verbose=False)
    b = Phi4Model(3, seed=7).fit(data, epochs=3, mcmc_steps=20, verbose=False)
    assert np.array_equal(a.W, b.W)


def test_predict_conditional_units_and_shapes():
    m = Phi4Model(4, seed=0)
    out = m.predict_conditional({0: 0.1}, target_idx=3, n_samples=300, burn=20)
    assert out.shape == (300,)
    out2 = m.predict_conditional({0: 0.1}, target_idx=[2, 3], n_samples=300, burn=20)
    assert out2.shape == (300, 2)
    with pytest.raises(ValueError):
        m.predict_conditional({3: 0.1}, target_idx=3)
    rets = np.random.default_rng(0).normal(0, 0.02, (100, 4))
    sc = Scaler("absmax").fit(rets)
    r = m.predict_conditional({0: 0.01}, target_idx=3, n_samples=300, burn=20, scaler=sc)
    assert np.abs(r).max() <= 1.5 * sc.absmax_[3] + 1e-12   # back in return units


def test_forecast_next_day_checks_length():
    m = Phi4Model(6, seed=0)
    assert m.forecast_next_day(np.zeros(5), n_samples=50, burn=5).shape == (50,)
    with pytest.raises(ValueError):
        m.forecast_next_day(np.zeros(6))
