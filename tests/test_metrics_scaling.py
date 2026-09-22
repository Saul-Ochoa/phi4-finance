import numpy as np
import pytest

from phi4finance.metrics import market_kurtosis, market_mean, binarize, susceptibility
from phi4finance.scaling import fit_exponent, scaling_analysis


def test_market_stats():
    X = np.random.default_rng(0).normal(0, 1, (5, 20000))
    assert np.allclose(market_kurtosis(X), 3.0, atol=0.15)
    assert market_mean(X).shape == (5,)
    assert set(np.unique(binarize([-2.0, 0.0, 3.0]))) == {-1.0, 0.0, 1.0}
    assert susceptibility(np.ones((10, 3))) == 0.0


def test_fit_exponent_recovers_power_law():
    V = np.array([16, 32, 48, 64])
    k, err = fit_exponent(V, 3.0 * V**-0.96)
    assert np.isclose(k, -0.96) and err < 1e-8
    with pytest.warns(UserWarning):
        assert np.isnan(fit_exponent(V, [1, -1, 1, 1])[0])


def test_scaling_analysis_smoke():
    X = np.random.default_rng(0).uniform(-0.5, 0.5, (60, 8))
    r = scaling_analysis(X, volumes=(3, 4, 6), n_iter=2, epochs=3, mcmc_steps=20,
                         statistic="absmean", verbose=False)
    assert r.volumes.tolist() == [3, 4, 6, 8]
    assert np.isfinite(r.k_w) and np.isfinite(r.k_a)
