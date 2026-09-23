import numpy as np
import pandas as pd
import pytest

from phi4finance import Phi4Model
from phi4finance.risk import coupling_matrix, empirical_stress, node_strength, stress_matrix, var_es


@pytest.fixture(scope="module")
def gaussian_fit():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(4000, 1))
    X = 0.15 * (0.8 * f + rng.normal(size=(4000, 4)))            # one common factor
    m = Phi4Model(4, mu_global=False, lam_global=False, seed=0).fit(X, method="pl", verbose=False)
    return X, m


def test_coupling_matrix_matches_partial_correlation(gaussian_fit):
    X, m = gaussian_fit
    th = np.linalg.inv(np.cov(X.T))
    pc = -th / np.sqrt(np.outer(np.diag(th), np.diag(th)))
    C = coupling_matrix(m, list("ABCD"))
    iu = np.triu_indices(4, 1)
    assert np.allclose(C.values[iu], pc[iu], atol=0.03)
    assert (np.diag(C.values) == 1).all()
    assert node_strength(C).shape == (4,) and (node_strength(C) > 0).all()


def test_stress_matrix_matches_gaussian_regression(gaussian_fit):
    X, m = gaussian_fit
    shock = -2.0 * X.std(0)
    S = stress_matrix(m, shock, n_samples=6000, burn=100)
    Cv = np.cov(X.T)
    for i in range(4):
        expected = Cv[:, i] / Cv[i, i] * shock[i]                   # E[x_j | x_i] for a Gaussian
        j = [k for k in range(4) if k != i]
        assert np.allclose(S.values[i, j], expected[j], atol=0.03)
    assert np.allclose(np.diag(S.values), shock)


def test_empirical_stress_and_var():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(5000, 3)), columns=list("xyz"))
    X["y"] = 0.5 * X["x"] + np.sqrt(0.75) * X["y"]
    S, shock = empirical_stress(X, q=0.05)
    assert shock["x"] < -1.8 and abs(S.loc["x", "y"] - 0.5 * shock["x"]) < 0.15
    v, es = var_es(rng.normal(size=100000), 0.95)
    assert abs(v - 1.645) < 0.03 and es > v
