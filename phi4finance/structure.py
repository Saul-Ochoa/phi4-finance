"""Parameter tying: which couplings share one value.

A ``Tying`` maps every pair (i, j) to a coupling group (or to -1: coupling
fixed at zero) and every site to a site group used by a_i, mu_i, lambda_i.
Estimators fit one parameter per group, so structure imposed by the problem
(stationarity in time, one set of biases per stock) cuts the parameter count
and the variance of the fit.

* ``Tying.free(V)``: every pair and site its own parameter (the paper's model).
* ``Tying.toeplitz(V)``: one stock's lags; w_ij depends only on |i - j|.
* ``Tying.lagged(n_assets, n_lags)``: several stocks over n_lags past days
  plus the current day. The coupling between stock a on day s and stock b on
  day t depends only on (a, b, t - s): same-day pairs are symmetric in (a, b);
  across days the direction matters (a leads b is not b leads a). This is the
  block-Toeplitz structure implied by stationarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tying:
    w_groups: np.ndarray          # (V, V) int, symmetric, -1 = fixed at zero
    site_groups: np.ndarray       # (V,) int
    w_labels: list = field(default_factory=list)
    site_labels: list = field(default_factory=list)

    def __post_init__(self):
        self.w_groups = np.asarray(self.w_groups, dtype=int)
        self.site_groups = np.asarray(self.site_groups, dtype=int)
        V = self.site_groups.size
        if self.w_groups.shape != (V, V) or not np.array_equal(self.w_groups, self.w_groups.T):
            raise ValueError("w_groups must be a symmetric (V, V) integer matrix")
        np.fill_diagonal(self.w_groups, -1)
        self.V = V
        self.iu = np.triu_indices(V, 1)
        self.pair_groups = self.w_groups[self.iu]
        self._mask = self.pair_groups >= 0
        self.n_w = int(self.pair_groups.max() + 1) if self._mask.any() else 0
        self.n_site = int(self.site_groups.max() + 1)
        if set(np.unique(self.site_groups)) != set(range(self.n_site)):
            raise ValueError("site_groups must use every id 0..n-1")
        used = np.unique(self.pair_groups[self._mask])
        if used.size != self.n_w:
            raise ValueError("coupling groups must use every id 0..n-1")

    # ---------------------------------------------------------------- maps
    def expand_w(self, theta) -> np.ndarray:
        W = np.zeros((self.V, self.V))
        vals = np.zeros(self.pair_groups.size)
        vals[self._mask] = np.asarray(theta, float)[self.pair_groups[self._mask]]
        W[self.iu] = vals
        return W + W.T

    def reduce_w(self, M, how: str = "sum") -> np.ndarray:
        """Per-group sum (gradients) or mean (projection) of the upper-triangle entries of M."""
        v = np.asarray(M, float)[self.iu][self._mask]
        g = self.pair_groups[self._mask]
        s = np.bincount(g, weights=v, minlength=self.n_w)
        if how == "mean":
            s = s / np.maximum(np.bincount(g, minlength=self.n_w), 1)
        return s

    def expand_site(self, theta) -> np.ndarray:
        return np.asarray(theta, float)[self.site_groups]

    def reduce_site(self, v, how: str = "sum") -> np.ndarray:
        s = np.bincount(self.site_groups, weights=np.asarray(v, float), minlength=self.n_site)
        if how == "mean":
            s = s / np.bincount(self.site_groups, minlength=self.n_site)
        return s

    def project(self, W, a, mu, lam):
        """Closest tied parameters (group means)."""
        return (self.expand_w(self.reduce_w(W, "mean")), self.expand_site(self.reduce_site(a, "mean")),
                self.expand_site(self.reduce_site(mu, "mean")), self.expand_site(self.reduce_site(lam, "mean")))

    @property
    def is_free(self) -> bool:
        return self.n_w == len(self.pair_groups) and self.n_site == self.V

    # ---------------------------------------------------------------- constructors
    @classmethod
    def free(cls, V: int) -> "Tying":
        G = -np.ones((V, V), dtype=int)
        iu = np.triu_indices(V, 1)
        G[iu] = np.arange(len(iu[0]))
        G = np.where(G >= 0, G, G.T)
        return cls(G, np.arange(V), [f"w{i},{j}" for i, j in zip(*iu)], [f"site{i}" for i in range(V)])

    @classmethod
    def toeplitz(cls, V: int, max_lag=None) -> "Tying":
        """One series embedded over V consecutive days: w_ij = w(|i - j|); one site group."""
        i, j = np.indices((V, V))
        d = np.abs(i - j)
        G = d - 1
        if max_lag is not None:
            G[d > max_lag] = -1
        np.fill_diagonal(G, -1)
        n = int(G.max() + 1)
        return cls(G, np.zeros(V, dtype=int), [f"lag{k + 1}" for k in range(n)], ["all"])

    @classmethod
    def lagged(cls, n_assets: int, n_lags: int, target=None, max_lag=None, names=None) -> "Tying":
        """Sites for ``n_lags`` past days plus the current day of ``n_assets``
        stocks, ordered day by day (oldest first) and stock by stock inside a
        day, as produced by ``lag_embed_panel``. With ``target`` = k, the
        current day keeps only stock k (the forecasting set-up)."""
        K = n_assets
        names = list(names) if names is not None else [f"x{k}" for k in range(K)]
        sites = [(s, a) for s in range(n_lags + 1) for a in range(K)
                 if s < n_lags or target is None or a == target]
        V = len(sites)
        keys, G = {}, -np.ones((V, V), dtype=int)
        for p in range(V):
            for q in range(p + 1, V):
                (s, a), (t, b) = sites[p], sites[q]          # s <= t by construction
                d = t - s
                if max_lag is not None and d > max_lag:
                    continue
                key = ("same", min(a, b), max(a, b)) if d == 0 else ("lead", a, b, d)
                G[p, q] = G[q, p] = keys.setdefault(key, len(keys))
        labels = [None] * len(keys)
        for key, g in keys.items():
            labels[g] = (f"{names[key[1]]}~{names[key[2]]} same day" if key[0] == "same"
                         else f"{names[key[1]]}(t-{key[3]})->{names[key[2]]}(t)")
        return cls(G, np.array([a for _, a in sites]), labels, names)


def lag_embed_panel(Z, n_lags: int, target=None) -> np.ndarray:
    """Rows of ``n_lags`` past days of every column of Z (T, K) followed by the
    current day (all columns, or only column ``target``), in the site order of
    ``Tying.lagged``. Returns (T - n_lags, K * n_lags + (K or 1))."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    T, K = Z.shape
    if T <= n_lags:
        raise ValueError(f"need more than n_lags={n_lags} rows")
    past = np.stack([Z[s:T - n_lags + s] for s in range(n_lags)], axis=1).reshape(T - n_lags, n_lags * K)
    cur = Z[n_lags:] if target is None else Z[n_lags:, [target]]
    return np.hstack([past, cur])
