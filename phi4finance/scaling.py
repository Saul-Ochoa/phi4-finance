"""Finite-size scaling of the learned couplings (Section 3.3, Appendix A.2).

    <w_ij> ~ V^{k_w},   <a_i> ~ V^{k_a}                                  (eq. 8)

Procedure of Appendix A.2: train the full V-stock model with global mu and
lambda, freeze those two values, then retrain only W and a on random subsets of
V' stocks (drawn without replacement), repeating each volume several times.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import linregress

from .model import Phi4Model


@dataclass
class ScalingResult:
    volumes: np.ndarray
    w_mean: np.ndarray
    w_sem: np.ndarray
    a_mean: np.ndarray
    a_sem: np.ndarray
    k_w: float
    k_w_err: float
    k_a: float
    k_a_err: float
    mu_global: float
    lam_global: float
    statistic: str
    raw: dict = field(default_factory=dict, repr=False)

    def exponents(self, statistic: str = "mean") -> dict:
        """k_w, k_a (and standard errors) for either statistic, from ``raw``."""
        key = {"mean": ("w", "a"), "absmean": ("w_abs", "a_abs")}[statistic]
        vols = np.array(sorted(self.raw))
        out = {}
        for name, k in zip(("w", "a"), key):
            vals = np.array([np.mean(self.raw[v][k]) for v in vols])
            out[f"k_{name}"], out[f"k_{name}_err"] = fit_exponent(vols, vals)
            out[f"{name}_values"] = vals
        out["volumes"] = vols
        return out


def fit_exponent(volumes, values):
    """Least-squares slope of log(values) vs log(volumes) and its standard error.

    Returns (nan, nan) with a warning if any value is <= 0 (a power law in a
    signed mean is only defined while the mean keeps one sign).
    """
    v = np.asarray(volumes, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(v) < 3:
        raise ValueError("need at least 3 volumes to fit an exponent with an error")
    if np.any(y <= 0):
        warnings.warn("non-positive mean coupling; exponent undefined (try statistic='absmean')")
        return float("nan"), float("nan")
    r = linregress(np.log(v), np.log(y))
    return float(r.slope), float(r.stderr)


def _stat(x, statistic):
    return float(np.mean(x) if statistic == "mean" else np.mean(np.abs(x)))


def scaling_analysis(full_returns, volumes=(16, 32, 48), n_iter=(100, 40, 40),
                     method: str = "pl", epochs: int = 200, mcmc_steps: int = 100,
                     lr: float = 5e-3, l2: float = 0.0, n_grid: int = 201, statistic: str = "mean",
                     seed: int = 0, verbose: bool = True) -> ScalingResult:
    """Scaling exponents k_w and k_a.

    Parameters
    ----------
    full_returns : array (N, V_full) in model units.
    volumes : subset sizes V' < V_full. The full volume is added automatically.
    n_iter : random subsets per volume (int or one per volume). The paper uses
        100, 40, 40 for V' = 16, 32, 48.
    method : estimator passed to ``Phi4Model.fit`` (``"pl"`` or ``"ml"``);
        ``epochs``, ``mcmc_steps`` and ``lr`` only apply to ``"ml"``.
    statistic : ``"mean"`` is the paper's signed average <w_ij> (upper
        triangle, i < j) and <a_i>; ``"absmean"`` averages magnitudes.
    """
    if statistic not in ("mean", "absmean"):
        raise ValueError("statistic must be 'mean' or 'absmean'")
    X = np.asarray(full_returns, dtype=float)
    V_full = X.shape[1]
    volumes = [int(v) for v in volumes]
    if any(v >= V_full or v < 2 for v in volumes):
        raise ValueError(f"volumes must be in [2, {V_full})")
    n_iter = [int(n_iter)] * len(volumes) if np.isscalar(n_iter) else [int(n) for n in n_iter]
    if len(n_iter) != len(volumes):
        raise ValueError("n_iter must be an int or have one entry per volume")
    rng = np.random.default_rng(seed)

    base = Phi4Model(V_full, lr=lr, mu_global=True, lam_global=True, seed=seed)
    fit_kw = {"method": method, "epochs": epochs, "mcmc_steps": mcmc_steps, "l2": l2, "n_grid": n_grid}
    base.fit(X, verbose=verbose, **fit_kw)
    mu_g, lam_g = float(base.mu.mean()), float(base.lam.mean())
    iu_full = np.triu_indices(V_full, 1)

    def stats(W, a, iu):
        return {"w": _stat(W[iu], "mean"), "a": _stat(a, "mean"),
                "w_abs": _stat(W[iu], "absmean"), "a_abs": _stat(a, "absmean")}

    wk, ak = ("w", "a") if statistic == "mean" else ("w_abs", "a_abs")
    raw = {V_full: {k: [v] for k, v in stats(base.W, base.a, iu_full).items()}}
    for V, n in zip(volumes, n_iter):
        iu = np.triu_indices(V, 1)
        acc = {"w": [], "a": [], "w_abs": [], "a_abs": []}
        for _ in range(n):
            idx = rng.choice(V_full, V, replace=False)
            m = Phi4Model(V, lr=lr, mu_global=True, lam_global=True,
                          seed=int(rng.integers(2**31)), freeze=("mu", "lam"))
            m.mu[:] = mu_g
            m.lam[:] = lam_g
            m.fit(X[:, idx], verbose=False, **fit_kw)
            for k, v in stats(m.W, m.a, iu).items():
                acc[k].append(v)
        raw[V] = acc
        if verbose:
            print(f"V={V:4d}  <w>={np.mean(acc['w']):+.5f}  <a>={np.mean(acc['a']):+.5f}  "
                  f"<|w|>={np.mean(acc['w_abs']):.5f}  <|a|>={np.mean(acc['a_abs']):.5f}  (n={n})")

    vols = np.array(sorted(raw))
    agg = lambda key, f: np.array([f(raw[v][key]) for v in vols])
    sem = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    w_mean, a_mean = agg(wk, np.mean), agg(ak, np.mean)
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        k_w, k_w_err = fit_exponent(vols, w_mean)
        k_a, k_a_err = fit_exponent(vols, a_mean)
    return ScalingResult(vols, w_mean, agg(wk, sem), a_mean, agg(ak, sem),
                         k_w, k_w_err, k_a, k_a_err, mu_g, lam_g, statistic, raw)
