"""phi^4 quantum field-theoretic machine learning (Bachtis et al. 2026, Section 2).

Two estimators are available through ``Phi4Model.fit``:

* ``method="pl"`` (default): maximum pseudo-likelihood, deterministic, no MCMC
  (see ``phi4finance.estimators``).
* ``method="ml"``: the paper's maximum likelihood (eq. 5) with gradient-ASCENT
  updates and persistent MCMC chains for the model expectations:

    w_ij  += lr * (<phi_i phi_j>_data - <phi_i phi_j>_model)
    a_i   += lr * (<phi_i>_data       - <phi_i>_model)
    mu_i  += lr * (<phi_i^2>_model    - <phi_i^2>_data)
    lam_i += lr * (<phi_i^4>_model    - <phi_i^4>_data)

``fit`` continues from the current couplings, so ``fit(method="pl")`` followed
by ``fit(method="ml")`` refines a pseudo-likelihood solution by likelihood.
"""
from __future__ import annotations

import numpy as np

from .estimators import fit_pseudolikelihood, penalty_weights
from .inference import ConditionalDistribution, exact_conditional
from .sampler import SAMPLERS
from .structure import Tying

try:  # optional progress bar
    from tqdm import trange as _trange
except ImportError:  # pragma: no cover
    _trange = None

_GROUPS = ("W", "a", "mu", "lam")


class Phi4Model:
    """Disordered phi^4 theory on a complete graph with V sites.

    Parameters
    ----------
    n_stocks : number of sites V (stocks, or lags in the forecasting set-up).
    lr : learning rate of ``method="ml"``.
    mu_global, lam_global : share a single mu (lambda) across sites, as in the
        scaling analysis of Appendix A.2.
    proposal_range : width r of the support [-r/2, r/2] (paper: 3).
    seed : seed of this model's own random generator.
    lam_min : lower bound kept on lambda so the quartic term stays confining.
    freeze : parameter groups never updated by ``fit`` (subset of
        {"W", "a", "mu", "lam"}).
    sampler : ``"metropolis"`` (paper) or ``"heatbath"`` (exact Gibbs).
    n_chains : parallel MCMC chains used by ``fit(method="ml")`` and sampling.
    tying : optional ``Tying`` saying which couplings / site parameters share a
        value (e.g. ``Tying.toeplitz(V)`` for one stock's lags,
        ``Tying.lagged(K, L)`` for K stocks over L days). Default: all free.
    """

    def __init__(self, n_stocks: int, lr: float = 5e-3, mu_global: bool = False,
                 lam_global: bool = True, proposal_range: float = 3.0, seed=0,
                 lam_min: float = 1e-4, freeze=(), sampler: str = "metropolis",
                 n_chains: int = 32, tying=None):
        if n_stocks < 1:
            raise ValueError("n_stocks must be >= 1")
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {list(SAMPLERS)}")
        self.V = int(n_stocks)
        self.lr = float(lr)
        self.mu_global = mu_global
        self.lam_global = lam_global
        self.proposal_range = float(proposal_range)
        self.lam_min = float(lam_min)
        self.freeze = set(freeze)
        unknown = self.freeze - set(_GROUPS)
        if unknown:
            raise ValueError(f"unknown parameter groups in freeze: {unknown}")
        self.sampler_kind = sampler
        self.n_chains = int(n_chains)
        self.rng = np.random.default_rng(seed)

        W = self.rng.normal(0.0, 0.01, (self.V, self.V))
        self.W = (W + W.T) / 2.0
        np.fill_diagonal(self.W, 0.0)
        self.mu = np.full(self.V, 0.5)
        self.lam = np.full(self.V, 0.5)
        self.a = np.zeros(self.V)
        self.tying = tying if tying is not None else Tying.free(self.V)
        if self.tying.V != self.V:
            raise ValueError(f"tying has V={self.tying.V}, model has V={self.V}")
        if not self.tying.is_free:
            self.W, self.a, self.mu, self.lam = self.tying.project(self.W, self.a, self.mu, self.lam)
        self.history: list[dict] = []
        self._chain = None

    # ------------------------------------------------------------ helpers
    def _sym(self):
        self.W = (self.W + self.W.T) / 2.0
        np.fill_diagonal(self.W, 0.0)

    def sampler(self, n_chains=None):
        cls = SAMPLERS[self.sampler_kind]
        return cls(self.W, self.mu, self.lam, self.a, proposal_range=self.proposal_range,
                   rng=self.rng, n_chains=n_chains or self.n_chains)

    def params(self) -> dict:
        return {"W": self.W.copy(), "a": self.a.copy(), "mu": self.mu.copy(), "lam": self.lam.copy()}

    @staticmethod
    def _moments(x):
        n = len(x)
        m = x.mean(axis=1)  # magnetisation per configuration (eq. 20)
        return {"phi": x.mean(0), "phi2": (x**2).mean(0), "phi4": (x**4).mean(0),
                "phiphi": x.T @ x / n, "m": m.mean(), "chi": m.var()}

    def gradients(self, data_batch, model_samples) -> dict:
        """Log-likelihood gradients (ascent directions) from a data batch and
        model samples, both of shape (n, V)."""
        q, p = self._moments(np.asarray(data_batch, float)), self._moments(np.asarray(model_samples, float))
        gW = q["phiphi"] - p["phiphi"]
        np.fill_diagonal(gW, 0.0)
        return {"W": gW, "a": q["phi"] - p["phi"], "mu": p["phi2"] - q["phi2"],
                "lam": p["phi4"] - q["phi4"], "_q": q, "_p": p}

    def _step(self, g, l2=0.0, scale=None):
        t = self.tying
        tie = (lambda v: t.expand_site(t.reduce_site(v))) if not t.is_free else (lambda v: v)
        if "W" not in self.freeze:
            S2 = 1.0 if scale is None else np.outer(scale, scale) ** 2
            gW = g["W"] - 2.0 * l2 * S2 * self.W
            if not t.is_free:
                gW = t.expand_w(t.reduce_w(gW))
            self.W += self.lr * gW
            self._sym()
        if "a" not in self.freeze:
            self.a += self.lr * tie(g["a"])
        if "mu" not in self.freeze:
            self.mu += self.lr * (g["mu"].mean() if self.mu_global else tie(g["mu"]))
        if "lam" not in self.freeze:
            self.lam += self.lr * (g["lam"].mean() if self.lam_global else tie(g["lam"]))
            np.maximum(self.lam, self.lam_min, out=self.lam)

    @staticmethod
    def _check_data(data, V):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != V:
            raise ValueError(f"data must have shape (N, {V}), got {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError("data contains NaN or inf")
        return data

    # ------------------------------------------------------------ training
    def fit(self, data, method: str = "pl", epochs: int = 300, mcmc_steps: int = 100,
            batch_size=None, burn=None, persistent: bool = True, l2: float = 0.0,
            penalty_scale: str = "std", n_grid: int = 201, maxiter: int = 500,
            verbose: bool = True):
        """Fit the couplings to ``data`` of shape (N, V) in model units.

        Parameters
        ----------
        method : ``"pl"`` (pseudo-likelihood, default) or ``"ml"`` (MCMC).
        l2 : penalty l2 * sum_{i<j} (s_i s_j w_ij)^2 (both methods).
        penalty_scale : ``"std"`` (default) sets s_i to the data's standard
            deviation, so ``l2`` does not depend on how the data were scaled;
            ``"none"`` sets s_i = 1 (the v0.4 penalty).
        n_grid, maxiter : quadrature cells per site and L-BFGS iterations (``"pl"``).
        epochs : gradient steps (``"ml"``).
        mcmc_steps : sweeps of each chain per epoch, burn-in included (``"ml"``).
        batch_size : data rows per epoch; ``None`` (default) or >= N is full batch.
        burn : sweeps discarded per epoch; default ``mcmc_steps // 3`` when a
            chain starts fresh and 0 when it continues from the last epoch.
        persistent : keep the ``n_chains`` chains across epochs (Persistent
            Contrastive Divergence) instead of restarting them.

        ``self.history`` receives one record per L-BFGS iteration (``"pl"``) or
        per epoch (``"ml"``: data vs model magnetisation and chi as in Fig. 7,
        mean |<phi_i phi_j>| gap and acceptance rate).
        """
        data = self._check_data(data, self.V)
        if method == "pl":
            self.fit_result_ = fit_pseudolikelihood(self, data, l2=l2, n_grid=n_grid, maxiter=maxiter,
                                                    penalty_scale=penalty_scale)
            self._chain = None
            if verbose:
                r = self.fit_result_
                print(f"pseudo-likelihood: {r.nit} iterations, -PL/N = {r.fun:.5f}, {r.message}")
            return self
        if method != "ml":
            raise ValueError("method must be 'pl' or 'ml'")

        N = data.shape[0]
        scale = penalty_weights(data, penalty_scale)
        bs = N if batch_size is None or batch_size >= N else int(batch_size)
        it = _trange(epochs) if (verbose and _trange is not None) else range(epochs)
        for epoch in it:
            batch = data if bs == N else data[self.rng.choice(N, bs, replace=False)]
            s = self.sampler()
            fresh = not (persistent and self._chain is not None and self._chain.shape == (s.n_chains, self.V))
            b = (mcmc_steps // 3 if fresh else 0) if burn is None else int(burn)
            if mcmc_steps - b < 1:
                raise ValueError("mcmc_steps must exceed burn")
            samples = s.sample(n_samples=(mcmc_steps - b) * s.n_chains, burn=b,
                               init=None if fresh else self._chain)
            self._chain = s.last_state
            g = self.gradients(batch, samples)
            self._step(g, l2=l2, scale=scale)
            q, p = g["_q"], g["_p"]
            rec = {"method": "ml", "epoch": epoch, "m_data": q["m"], "m_model": p["m"],
                   "chi_data": q["chi"], "chi_model": p["chi"],
                   "phiphi_gap": float(np.abs(g["W"]).mean()),
                   "acceptance": s.acceptance_rate}
            self.history.append(rec)
            if verbose and _trange is not None and epoch % 10 == 0:
                it.set_postfix({"m_q": f"{q['m']:.3f}", "m_p": f"{p['m']:.3f}",
                                "gap": f"{rec['phiphi_gap']:.4f}"})
        return self

    # ------------------------------------------------------------ sampling / inference
    def sample(self, n_samples: int = 2000, burn: int = 500, thin: int = 1) -> np.ndarray:
        """Unconditional configurations (n_samples, V) in model units."""
        return self.sampler().sample(n_samples=n_samples, burn=burn, thin=thin)

    @staticmethod
    def _affine_of(scaler, col):
        shift = float(np.asarray(scaler.inverse_transform(0.0, cols=col)))
        scale = float(np.asarray(scaler.inverse_transform(1.0, cols=col))) - shift
        return scale, shift

    def conditional_distribution(self, known_dict: dict, target_idx: int, scaler=None,
                                 n_grid: int = 801) -> ConditionalDistribution:
        """Exact p(phi_target | all other sites) by 1-D quadrature (no MCMC).

        ``known_dict`` must give every site except ``target_idx``. With
        ``scaler`` the known values are in return units and the distribution
        is returned in return units.
        """
        t = int(target_idx)
        missing = set(range(self.V)) - set(int(k) for k in known_dict) - {t}
        if missing:
            raise ValueError(f"exact conditional needs every other site; missing {sorted(missing)[:5]}"
                             f"{'...' if len(missing) > 5 else ''} (use predict_conditional for MCMC)")
        phi = np.zeros(self.V)
        for k, v in known_dict.items():
            k = int(k)
            phi[k] = float(scaler.transform(v, cols=k)) if scaler is not None else float(v)
        dist = exact_conditional(self.W, self.mu, self.lam, self.a, phi, t,
                                 proposal_range=self.proposal_range, n_grid=n_grid)
        if scaler is not None:
            dist = dist.affine(*self._affine_of(scaler, t))
        return dist

    def predict_conditional(self, known_dict: dict, target_idx, n_samples: int = 2000,
                            burn: int = 500, thin: int = 1, scaler=None,
                            method: str = "auto") -> np.ndarray:
        """Sample p(phi_target | phi_known) (eqs. 9, 11, 21).

        Sites that are neither known nor targets are marginalised.

        method : ``"exact"`` draws from the 1-D quadrature (needs every other
            site known and one target), ``"mcmc"`` runs the model's sampler,
            ``"auto"`` picks ``"exact"`` whenever it applies.
        scaler : optional fitted ``Scaler``; known values and returned samples
            are then in return units (site index = scaler column, ignored for a
            scaler fitted on one series).

        Returns an array (n_samples,) for an int target, (n_samples, k) for k targets.
        """
        single = np.isscalar(target_idx)
        targets = [int(target_idx)] if single else [int(t) for t in target_idx]
        known = {int(k): float(v) for k, v in known_dict.items()}
        clash = set(targets) & set(known)
        if clash:
            raise ValueError(f"sites {sorted(clash)} are both known and targets")
        exact_ok = single and len(known) == self.V - 1
        if method not in ("auto", "exact", "mcmc"):
            raise ValueError("method must be 'auto', 'exact' or 'mcmc'")
        if method == "exact" and not exact_ok:
            raise ValueError("method='exact' needs one target and every other site known")

        if exact_ok and method in ("auto", "exact"):
            dist = self.conditional_distribution(known, targets[0], scaler=scaler)
            return dist.sample(n_samples, rng=self.rng)

        if scaler is not None:
            known = {k: float(scaler.transform(v, cols=k)) for k, v in known.items()}
        samples = self.sampler().sample(n_samples=n_samples, burn=burn, thin=thin, fixed=known)
        out = samples[:, targets]
        if scaler is not None:
            out = np.asarray(scaler.inverse_transform(out, cols=targets))
        return out[:, 0] if single else out

    def _history_known(self, history):
        history = np.asarray(history, dtype=float).ravel()
        if self.V != len(history) + 1:
            raise ValueError(f"model has V={self.V}; history must have {self.V - 1} values, got {len(history)}")
        return {i: history[i] for i in range(len(history))}

    def forecast_distribution(self, history, scaler=None, n_grid: int = 801) -> ConditionalDistribution:
        """Exact p(phi_next | phi_0, ..., phi_-(V-2)) (eq. 11). ``history``
        holds the V-1 most recent returns, oldest first (as ``lag_embed``)."""
        return self.conditional_distribution(self._history_known(history), self.V - 1,
                                             scaler=scaler, n_grid=n_grid)

    def forecast_next_day(self, history, n_samples: int = 3000, burn: int = 500,
                          thin: int = 1, scaler=None, method: str = "auto") -> np.ndarray:
        """Samples of the next return given the V-1 previous ones (eq. 11)."""
        return self.predict_conditional(self._history_known(history), target_idx=self.V - 1,
                                        n_samples=n_samples, burn=burn, thin=thin,
                                        scaler=scaler, method=method)

    def __repr__(self):
        return (f"Phi4Model(V={self.V}, sampler={self.sampler_kind!r}, n_chains={self.n_chains}, "
                f"tying={'free' if self.tying.is_free else f'{self.tying.n_w} coupling groups'}, "
                f"mu_global={self.mu_global}, lam_global={self.lam_global}, "
                f"history={len(self.history)})")
