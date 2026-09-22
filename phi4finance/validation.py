"""Choosing the L2 penalty on held-out rows (Appendix A.3 uses 10 randomly
excluded training rows and early stopping; with pseudo-likelihood the L2
penalty plays the role of early stopping)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import Phi4Model


def select_l2(X, target_idx: int, l2_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0), n_val: int = 10,
              seed: int = 0, model_kw=None, fit_kw=None):
    """Pick ``l2`` by the MAE of the exact conditional mean of ``target_idx``
    on ``n_val`` randomly held-out rows of ``X`` (model units).

    Returns ``(best_l2, table)`` where ``table`` has one row per candidate.
    Refit on all rows with the chosen value afterwards.
    """
    X = np.asarray(X, dtype=float)
    N, V = X.shape
    if not 1 <= n_val < N:
        raise ValueError("n_val must be in [1, N)")
    rng = np.random.default_rng(seed)
    val = rng.choice(N, n_val, replace=False)
    train = np.setdiff1d(np.arange(N), val)
    model_kw = dict(model_kw or {}); fit_kw = dict(fit_kw or {})
    rows, prev = [], None
    for l2 in sorted(l2_grid, reverse=True):     # strong -> weak penalty, each fit warm-started
        m = Phi4Model(V, **model_kw)
        if prev is not None:
            m.W, m.a, m.mu, m.lam = prev.W.copy(), prev.a.copy(), prev.mu.copy(), prev.lam.copy()
        m.fit(X[train], method="pl", l2=l2, verbose=False, **fit_kw)
        prev = m
        preds = []
        for r in X[val]:
            known = {j: r[j] for j in range(V) if j != target_idx}
            preds.append(m.conditional_distribution(known, target_idx).mean())
        rows.append({"l2": l2, "val_mae": float(np.mean(np.abs(np.array(preds) - X[val, target_idx])))})
    table = pd.DataFrame(rows).sort_values("l2").reset_index(drop=True)
    return float(table.loc[table.val_mae.idxmin(), "l2"]), table
