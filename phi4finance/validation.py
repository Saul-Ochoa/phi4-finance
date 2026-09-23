"""Choosing the L2 penalty on held-out rows.

Appendix A.3 of the paper holds out 10 random training rows and uses early
stopping. With pseudo-likelihood the L2 penalty plays the role of early
stopping; ``select_l2`` picks it either on random held-out rows (the paper's
scheme) or by blocked cross-validation over contiguous folds, which is less
noisy and respects time order within each fold.

Since v0.5 the penalty is scale-free (``penalty_scale="std"``): it acts on the
couplings of standardised data, so a value means the same thing whatever the
scaling of the inputs. How much penalty a problem needs still depends on the
ratio of parameters to rows (the 150-lag forecaster with 80 rows needs values
around 10^3-10^4), hence a wide default grid in decades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import Phi4Model

DEFAULT_L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def _score(m, rows, target_idx, metric):
    V = rows.shape[1]
    out = []
    for r in rows:
        d = m.conditional_distribution({j: r[j] for j in range(V) if j != target_idx}, target_idx)
        out.append(abs(d.mean() - r[target_idx]) if metric == "mae" else d.crps(r[target_idx]))
    return float(np.mean(out))


def select_l2(X, target_idx: int, l2_grid=DEFAULT_L2_GRID, n_val: int = 10, folds=None,
              metric: str = "mae", seed: int = 0, model_kw=None, fit_kw=None):
    """Pick ``l2`` by the held-out error of the exact conditional of
    ``target_idx`` (model units).

    Parameters
    ----------
    n_val : random held-out rows (used when ``folds`` is None).
    folds : number of contiguous blocks for blocked cross-validation; each
        block is held out once and the scores are averaged.
    metric : ``"mae"`` of the conditional mean or ``"crps"`` of the whole
        conditional distribution.

    Returns ``(best_l2, table)``. Refit on all rows with the chosen value.
    """
    X = np.asarray(X, dtype=float)
    N, V = X.shape
    if metric not in ("mae", "crps"):
        raise ValueError("metric must be 'mae' or 'crps'")
    if folds is None:
        if not 1 <= n_val < N:
            raise ValueError("n_val must be in [1, N)")
        val = np.random.default_rng(seed).choice(N, n_val, replace=False)
        splits = [(np.setdiff1d(np.arange(N), val), val)]
    else:
        if not 2 <= folds <= N // 2:
            raise ValueError("folds must be in [2, N/2]")
        blocks = np.array_split(np.arange(N), folds)
        splits = [(np.setdiff1d(np.arange(N), b), b) for b in blocks]
    model_kw = dict(model_kw or {}); fit_kw = dict(fit_kw or {})
    scores = {l2: [] for l2 in l2_grid}
    for train, val in splits:
        prev = None
        for l2 in sorted(l2_grid, reverse=True):     # strong -> weak penalty, warm-started
            m = Phi4Model(V, **model_kw)
            if prev is not None:
                m.W, m.a, m.mu, m.lam = prev.W.copy(), prev.a.copy(), prev.mu.copy(), prev.lam.copy()
            m.fit(X[train], method="pl", l2=l2, verbose=False, **fit_kw)
            prev = m
            scores[l2].append(_score(m, X[val], target_idx, metric))
    table = pd.DataFrame({"l2": list(scores), f"val_{metric}": [np.mean(v) for v in scores.values()]})
    table = table.sort_values("l2").reset_index(drop=True)
    return float(table.loc[table[f"val_{metric}"].idxmin(), "l2"]), table
