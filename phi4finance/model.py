"""phi^4 quantum field-theoretic machine learning (Bachtis et al. 2026, Section 2).

Training maximises the log-likelihood of p(phi) = exp(-S(phi)) / Z. From eq. 5,

    d ln p / d theta = < dS/d theta >_model - < dS/d theta >_data

which for the couplings of eq. 1 gives the gradient-ASCENT updates

    w_ij  += lr * (<phi_i phi_j>_data - <phi_i phi_j>_model)
    a_i   += lr * (<phi_i>_data       - <phi_i>_model)
    mu_i  += lr * (<phi_i^2>_model    - <phi_i^2>_data)
    lam_i += lr * (<phi_i^4>_model    - <phi_i^4>_data)

Model expectations are estimated with the Metropolis sampler.
"""
from __future__ import annotations

import numpy as np

from .sampler import MetropolisSampler

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
    lr : learning rate.
    mu_global, lam_global : share a single mu (lambda) across sites, as in the
        scaling analysis of Appendix A.2.
    proposal_range : width r of the uniform Metropolis proposal on [-r/2, r/2].
    seed : seed of this model's own random generator.
    lam_min : lower bound kept on lambda so the quartic term stays confining.
    freeze : parameter groups not updated by ``fit`` (subset of
        {"W", "a", "mu", "lam"}); the scaling analysis freezes mu and lam.
    """

    def __init__(self, n_stocks: int, lr: float = 5e-3, mu_global: bool = False,
                 lam_global: bool = True, proposal_range: float = 3.0, seed=0,
                 lam_min: float = 1e-4, freeze=()):
        if n_stocks < 1:
            raise ValueError("n_stocks must be >= 1")
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
        self.rng = np.random.default_rng(seed)

        W = self.rng.normal(0.0, 0.01, (self.V, self.V))
        self.W = (W + W.T) / 2.0
        np.fill_diagonal(self.W, 0.0)
        self.mu = np.full(self.V, 0.5)
        self.lam = np.full(self.V, 0.5)
        self.a = np.zeros(self.V)
        self.history: list[dict] = []
        self._chain = None

    # ------------------------------------------------------------ helpers
    def _sym(self):
        self.W = (self.W + self.W.T) / 2.0
        np.fill_diagonal(self.W, 0.0)

    def sampler(self) -> MetropolisSampler:
        return MetropolisSampler(self.W, self.mu, self.lam, self.a,
                                 proposal_range=self.proposal_range, rng=self.rng)

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

    def _step(self, g):
        if "W" not in self.freeze:
            self.W += self.lr * g["W"]
            self._sym()
        if "a" not in self.freeze:
            self.a += self.lr * g["a"]
        if "mu" not in self.freeze:
            self.mu += self.lr * (g["mu"].mean() if self.mu_global else g["mu"])
        if "lam" not in self.freeze:
            self.lam += self.lr * (g["lam"].mean() if self.lam_global else g["lam"])
            np.maximum(self.lam, self.lam_min, out=self.lam)

    # ------------------------------------------------------------ training
    def fit(self, data, epochs: int = 300, mcmc_steps: int = 1000, batch_size=None,
            burn=None, persistent: bool = True, verbose: bool = True):
        """Fit the couplings to ``data`` of shape (N, V) in model units.

        Parameters
        ----------
        epochs : gradient steps.
        mcmc_steps : Metropolis sweeps per epoch (burn-in included).
        batch_size : data rows per step; ``None`` (default) or >= N uses the
            full batch, which suits the paper's small data sets (80 rows).
        burn : sweeps discarded per epoch; default ``mcmc_steps // 3``.
        persistent : start each epoch's chain from the previous epoch's last
            state instead of a fresh random configuration.

        After fitting, ``self.history`` holds per-epoch diagnostics: data and
        model magnetisation <m> and fluctuation chi (Fig. 7) and the mean
        absolute gap in <phi_i phi_j>.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != self.V:
            raise ValueError(f"data must have shape (N, {self.V}), got {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError("data contains NaN or inf")
        N = data.shape[0]
        bs = N if batch_size is None or batch_size >= N else int(batch_size)
        burn = mcmc_steps // 3 if burn is None else int(burn)
        n_keep = mcmc_steps - burn
        if n_keep < 1:
            raise ValueError("mcmc_steps must exceed burn")

        it = _trange(epochs) if (verbose and _trange is not None) else range(epochs)
        for epoch in it:
            batch = data if bs == N else data[self.rng.choice(N, bs, replace=False)]
            s = self.sampler()
            init = self._chain if persistent else None
            samples = s.sample(n_samples=n_keep, burn=burn, init=init)
            self._chain = s.last_state
            g = self.gradients(batch, samples)
            self._step(g)
            q, p = g["_q"], g["_p"]
            rec = {"epoch": epoch, "m_data": q["m"], "m_model": p["m"],
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

    def predict_conditional(self, known_dict: dict, target_idx, n_samples: int = 2000,
                            burn: int = 500, thin: int = 1, scaler=None) -> np.ndarray:
        """Sample p(phi_target | phi_known) (eqs. 9, 11, 21).

        All sites that are neither known nor targets are sampled too and thus
        marginalised out.

        Parameters
        ----------
        known_dict : {site: value}.
        target_idx : int or list of ints.
        scaler : optional fitted ``Scaler``. If given, ``known_dict`` values are
            in return units and the samples are returned in return units; the
            site index is used as the scaler column (ignored for a scaler
            fitted on a single series).

        Returns
        -------
        array (n_samples,) for an int target, (n_samples, len(targets)) otherwise.
        """
        single = np.isscalar(target_idx)
        targets = [int(target_idx)] if single else [int(t) for t in target_idx]
        clash = set(targets) & set(known_dict)
        if clash:
            raise ValueError(f"sites {sorted(clash)} are both known and targets")
        known = {int(k): float(v) for k, v in known_dict.items()}
        if scaler is not None:
            known = {k: float(scaler.transform(v, cols=k)) for k, v in known.items()}
        samples = self.sampler().sample(n_samples=n_samples, burn=burn, thin=thin, fixed=known)
        out = samples[:, targets]
        if scaler is not None:
            out = np.asarray(scaler.inverse_transform(out, cols=targets))
        return out[:, 0] if single else out

    def forecast_next_day(self, history, n_samples: int = 3000, burn: int = 500,
                          thin: int = 1, scaler=None) -> np.ndarray:
        """Sample p(phi_next | phi_0, ..., phi_-(V-2)) (eq. 11).

        ``history`` holds the V-1 most recent returns in chronological order
        (oldest first), matching ``lag_embed``.
        """
        history = np.asarray(history, dtype=float).ravel()
        if self.V != len(history) + 1:
            raise ValueError(f"model has V={self.V}; history must have {self.V - 1} values, got {len(history)}")
        known = {i: history[i] for i in range(len(history))}
        return self.predict_conditional(known, target_idx=self.V - 1, n_samples=n_samples,
                                        burn=burn, thin=thin, scaler=scaler)

    def __repr__(self):
        return (f"Phi4Model(V={self.V}, lr={self.lr}, mu_global={self.mu_global}, "
                f"lam_global={self.lam_global}, epochs_trained={len(self.history)})")
