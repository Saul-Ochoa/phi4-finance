import numpy as np
import pytest

GRID = np.linspace(-1.5, 1.5, 601)


def exact_gibbs(W, mu, lam, a, n_chains=2000, sweeps=40, seed=0):
    """Independent reference sampler: heat-bath Gibbs drawing each site from its
    exact 1-D conditional on a grid over [-1.5, 1.5], vectorised over chains."""
    rng = np.random.default_rng(seed)
    V = len(a)
    mu = np.broadcast_to(mu, (V,)); lam = np.broadcast_to(lam, (V,))
    phi = rng.uniform(-1, 1, (n_chains, V))
    for _ in range(sweeps):
        for i in range(V):
            h = 2.0 * phi @ W[:, i] + a[i]
            ld = h[:, None] * GRID - mu[i] * GRID**2 - lam[i] * GRID**4
            p = np.exp(ld - ld.max(1, keepdims=True))
            cdf = np.cumsum(p, 1); cdf /= cdf[:, -1:]
            idx = (cdf < rng.random((n_chains, 1))).sum(1)
            phi[:, i] = GRID[np.minimum(idx, len(GRID) - 1)]
    return phi


@pytest.fixture(scope="session")
def truth():
    rng = np.random.default_rng(1)
    V = 5
    W = rng.normal(0, 0.4, (V, V)); W = (W + W.T) / 2; np.fill_diagonal(W, 0)
    mu, lam, a = np.full(V, 1.0), np.full(V, 0.8), rng.normal(0, 0.3, V)
    data = exact_gibbs(W, mu, lam, a, n_chains=4000)
    return {"W": W, "mu": mu, "lam": lam, "a": a, "data": data, "V": V}
