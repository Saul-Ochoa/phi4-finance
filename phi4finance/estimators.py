"""Pseudo-likelihood estimation (Besag) for the phi^4 couplings.

Instead of the full likelihood, whose gradient needs MCMC expectations
under the model, maximise the sum of one-site conditional log-likelihoods

    PL(theta) = (1/N) sum_n sum_i ln p(x_ni | x_n,-i)
    ln p(x_i | x_-i) = h_i x_i - mu_i x_i^2 - lam_i x_i^4 - ln Z_i(h_i),
    h_i = a_i + 2 sum_j w_ij x_j

Each Z_i is a 1-D integral over the truncated support, computed on a grid, so
PL and its exact gradient are deterministic and cheap. The estimator is
consistent for Markov random fields; with small samples an L2 penalty on W
(``l2``) is advisable.

Gradients, with r_ni = x_ni - E[phi | h_ni]:
    dPL/da_i    = mean_n r_ni
    dPL/dw_ij   = mean_n 2 (r_ni x_nj + r_nj x_ni)          (i < j)
    dPL/dmu_i   = mean_n (E[phi^2 | h_ni] - x_ni^2)
    dPL/dlam_i  = mean_n (E[phi^4 | h_ni] - x_ni^4)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .inference import site_grid


class _Packer:
    """Maps the free parameters of a Phi4Model to and from one flat vector."""

    def __init__(self, V, mu_global, lam_global, freeze):
        self.V = V
        self.iu = np.triu_indices(V, 1)
        self.groups = []
        sizes = {"W": len(self.iu[0]), "a": V, "mu": 1 if mu_global else V,
                 "lam": 1 if lam_global else V}
        start = 0
        for g in ("W", "a", "mu", "lam"):
            if g in freeze:
                continue
            self.groups.append((g, start, start + sizes[g]))
            start += sizes[g]
        self.size = start

    def pack(self, p):
        parts = []
        for g, s, e in self.groups:
            if g == "W":
                parts.append(p["W"][self.iu])
            elif e - s == 1:
                parts.append([np.mean(p[g])])
            else:
                parts.append(p[g])
        return np.concatenate(parts) if parts else np.empty(0)

    def unpack(self, x, base):
        p = {k: np.array(v, dtype=float) for k, v in base.items()}
        for g, s, e in self.groups:
            if g == "W":
                W = np.zeros((self.V, self.V))
                W[self.iu] = x[s:e]
                p["W"] = W + W.T
            else:
                p[g] = np.broadcast_to(x[s:e], (self.V,)).copy()
        return p

    def bounds(self, lam_min):
        b = []
        for g, s, e in self.groups:
            b += [(lam_min, None) if g == "lam" else (None, None)] * (e - s)
        return b


def neg_pseudo_loglik(p, X, grid, l2=0.0, chunk_elems=4_000_000):
    """Mean negative pseudo-log-likelihood per data row and its gradient
    (dict with keys W, a, mu, lam, each full-size)."""
    N, V = X.shape
    W, a, mu, lam = p["W"], p["a"], p["mu"], p["lam"]
    g1, g2, g4 = grid, grid**2, grid**4
    G = grid.size
    dx = float(grid[1] - grid[0])
    rows = max(1, int(chunk_elems // (V * G)))

    total = 0.0
    R_T_X = np.zeros((V, V))
    ga = np.zeros(V); gmu = np.zeros(V); glam = np.zeros(V)
    for s in range(0, N, rows):
        x = X[s:s + rows]
        H = a + 2.0 * (x @ W)                                        # (n, V)
        L = H[..., None] * g1 - mu[:, None] * g2 - lam[:, None] * g4  # (n, V, G)
        Lmax = L.max(axis=-1, keepdims=True)
        P = np.exp(L - Lmax)
        Z = P.sum(axis=-1, keepdims=True)
        P /= Z
        logZ = (Lmax + np.log(Z))[..., 0] + np.log(dx)   # midpoint rule: Z ≈ dx * sum
        x2, x4 = x**2, x**4
        total += float((H * x - mu * x2 - lam * x4 - logZ).sum())
        E1, E2, E4 = P @ g1, P @ g2, P @ g4
        r = x - E1
        R_T_X += r.T @ x
        ga += r.sum(0)
        gmu += (E2 - x2).sum(0)
        glam += (E4 - x4).sum(0)

    gW = 2.0 * (R_T_X + R_T_X.T)
    np.fill_diagonal(gW, 0.0)
    # dPL/dw_ij for the tied pair (i<j) is gW[i, j]; the penalty acts on each pair once
    f = -total / N + l2 * float((W[np.triu_indices(V, 1)] ** 2).sum())
    grads = {"W": -gW / N + 2.0 * l2 * W, "a": -ga / N, "mu": -gmu / N, "lam": -glam / N}
    return f, grads


def fit_pseudolikelihood(model, X, l2=0.0, n_grid=201, maxiter=500, tol=1e-7):
    """Fit ``model``'s free couplings to ``X`` (N, V) by maximum
    pseudo-likelihood with L-BFGS-B, starting from its current values.
    Respects ``mu_global``, ``lam_global``, ``freeze`` and ``lam_min``.
    Returns the scipy ``OptimizeResult``; per-iteration objective values
    are appended to ``model.history``.
    """
    X = np.asarray(X, dtype=float)
    packer = _Packer(model.V, model.mu_global, model.lam_global, model.freeze)
    if packer.size == 0:
        raise ValueError("every parameter group is frozen; nothing to fit")
    base = model.params()
    grid = site_grid(model.proposal_range, n_grid)
    if np.abs(X).max() > model.proposal_range / 2:
        raise ValueError("data fall outside the sampled support [-r/2, r/2]; "
                         "rescale the data or increase proposal_range")

    cache = {}

    def fun(x):
        p = packer.unpack(x, base)
        f, g = neg_pseudo_loglik(p, X, grid, l2=l2)
        cache["x"], cache["f"] = x.copy(), f
        flat = []
        for name, s, e in packer.groups:
            if name == "W":
                flat.append(g["W"][packer.iu])
            elif e - s == 1:
                flat.append([g[name].sum()])
            else:
                flat.append(g[name])
        return f, np.concatenate(flat)

    start = len(model.history)

    def callback(xk):
        f = cache["f"] if np.array_equal(cache.get("x"), xk) else fun(xk)[0]
        model.history.append({"method": "pl", "iter": len(model.history) - start, "neg_pl": f})

    res = minimize(fun, packer.pack(base), jac=True, method="L-BFGS-B",
                   bounds=packer.bounds(model.lam_min), callback=callback,
                   options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-6})
    p = packer.unpack(res.x, base)
    model.W, model.a, model.mu, model.lam = p["W"], p["a"], p["mu"], p["lam"]
    model._sym()
    return res
