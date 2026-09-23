"""Risk maps from a fitted phi^4 model.

* ``coupling_matrix``: w_ij / sqrt(mu_i mu_j), the model's partial correlation
  in the Gaussian limit (lambda = 0) and a scale-free view of the couplings.
* ``node_strength``: how connected each asset is (sum of |couplings|).
* ``stress_matrix``: E[phi_j | phi_i = shock_i] for every pair, by clamping one
  site and sampling the other V - 1 (conditional stress scenarios).
* ``empirical_stress``: the data counterpart, the average of x_j on the days
  when x_i is in its lower tail, so the model's scenarios can be checked.
* ``var_es``: historical value at risk and expected shortfall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def coupling_matrix(model, names=None) -> pd.DataFrame:
    """Scale-free couplings w_ij / sqrt(mu_i mu_j) (diagonal set to 1)."""
    mu = np.asarray(model.mu, float)
    if np.any(mu <= 0):
        raise ValueError("coupling_matrix needs mu_i > 0 for every site")
    C = model.W / np.sqrt(np.outer(mu, mu))
    np.fill_diagonal(C, 1.0)
    names = names if names is not None else range(model.V)
    return pd.DataFrame(C, index=names, columns=names)


def node_strength(C) -> pd.Series:
    """Sum over partners of |C_ij| (diagonal excluded)."""
    M = pd.DataFrame(C).copy()
    np.fill_diagonal(M.values, 0.0)
    return M.abs().sum(axis=1)


def empirical_stress(X, q: float = 0.05) -> tuple:
    """For each asset i, the days when X[:, i] is at or below its q-quantile.

    Returns (S, shock): S[i, j] = mean of X[:, j] on asset i's tail days
    (S[i, i] = the average shock itself), and ``shock`` = diag(S).
    """
    df = pd.DataFrame(X)
    S = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    for c in df.columns:
        tail = df[c] <= df[c].quantile(q)
        S.loc[c] = df.loc[tail].mean()
    return S, pd.Series(np.diag(S.values), index=df.columns)


def stress_matrix(model, shocks, names=None, n_samples: int = 3000, burn: int = 200) -> pd.DataFrame:
    """S[i, j] = E[phi_j | phi_i = shocks[i]] in model units, by MCMC over the
    other V - 1 sites (S[i, i] = shocks[i])."""
    shocks = np.asarray(shocks, float)
    V = model.V
    if shocks.shape != (V,):
        raise ValueError(f"shocks must have length {V}")
    S = np.empty((V, V))
    others = list(range(V))
    for i in range(V):
        targets = [j for j in others if j != i]
        samp = model.predict_conditional({i: shocks[i]}, targets, n_samples=n_samples, burn=burn, method="mcmc")
        S[i, targets] = samp.mean(axis=0)
        S[i, i] = shocks[i]
    names = names if names is not None else range(V)
    return pd.DataFrame(S, index=names, columns=names)


def var_es(x, level: float = 0.95) -> tuple:
    """Historical VaR and expected shortfall of returns ``x`` at ``level``,
    both reported as positive losses."""
    x = np.asarray(x, float)
    q = np.quantile(x, 1 - level)
    return float(-q), float(-x[x <= q].mean())
