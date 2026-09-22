"""MCMC samplers for the phi^4 lattice action (eq. 1).

    S(phi) = - sum_{i,j} w_ij phi_i phi_j + sum_i mu_i phi_i^2
             + sum_i lam_i phi_i^4 - sum_i a_i phi_i

Both samplers work on the bounded support [-r/2, r/2]^V with
r = ``proposal_range`` (r = 3 gives the paper's [-1.5, 1.5], Appendix A.1), so
they sample exp(-S) *truncated* to that hypercube.

Both run ``n_chains`` independent chains at once: one sweep updates site i in
every chain with a single vectorised numpy operation, then moves to site i+1.
The change in action for site i is local, O(V):

    dS = -h_i (new - old) + mu_i (new^2 - old^2) + lam_i (new^4 - old^4),
    h_i = a_i + 2 sum_j w_ij phi_j        (W symmetric, zero diagonal)
"""
from __future__ import annotations

import numpy as np


class _BaseSampler:
    def __init__(self, W, mu, lam, a, proposal_range: float = 3.0, rng=None, n_chains: int = 1):
        a = np.asarray(a, dtype=float)
        V = a.shape[0]
        W = np.array(W, dtype=float)
        if W.shape != (V, V):
            raise ValueError(f"W must be ({V}, {V}), got {W.shape}")
        if not np.allclose(W, W.T):
            raise ValueError("W must be symmetric")
        if n_chains < 1:
            raise ValueError("n_chains must be >= 1")
        np.fill_diagonal(W, 0.0)
        self.W = W
        self.mu = np.broadcast_to(np.asarray(mu, dtype=float), (V,)).copy()
        self.lam = np.broadcast_to(np.asarray(lam, dtype=float), (V,)).copy()
        self.a = a
        self.V = V
        self.n_chains = int(n_chains)
        self.proposal_range = float(proposal_range)
        self.half = self.proposal_range / 2.0
        self.rng = rng if rng is not None else np.random.default_rng()

    # ----------------------------------------------------------- action
    def action(self, phi):
        """S for one configuration (V,) or a batch (C, V)."""
        phi = np.asarray(phi, dtype=float)
        s = (-np.einsum("...i,ij,...j->...", phi, self.W, phi)
             + (phi**2) @ self.mu + (phi**4) @ self.lam - phi @ self.a)
        return float(s) if np.ndim(s) == 0 else s

    def local_field(self, phi, i: int):
        """h_i for one configuration (V,) or a batch (C, V)."""
        return self.a[i] + 2.0 * (np.asarray(phi) @ self.W[:, i])

    def delta_action(self, phi, i: int, new):
        old = np.asarray(phi)[..., i]
        h = self.local_field(phi, i)
        return (-h * (new - old) + self.mu[i] * (new**2 - old**2)
                + self.lam[i] * (new**4 - old**4))

    # ----------------------------------------------------------- driver
    def _sweep(self, phi, free):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _init(self, init, fixed):
        C, V = self.n_chains, self.V
        if init is None:
            lim = min(1.0, self.half)
            phi = self.rng.uniform(-lim, lim, size=(C, V))
        else:
            phi = np.array(init, dtype=float)
            if phi.shape == (V,):
                phi = np.tile(phi, (C, 1))
            if phi.shape != (C, V):
                raise ValueError(f"init must have shape ({V},) or ({C}, {V}), got {phi.shape}")
        for k, v in fixed.items():
            phi[:, k] = float(v)
        return phi

    def sample(self, n_samples: int = 1000, burn: int = 500, thin: int = 1,
               init=None, fixed: dict | None = None) -> np.ndarray:
        """Draw ``n_samples`` configurations from p(phi), or from
        p(phi_free | phi_fixed) when ``fixed`` = {site: value} is given.

        Every chain runs ``burn`` sweeps, then ``ceil(n_samples / n_chains)``
        recorded sweeps spaced ``thin`` apart; rows are interleaved across
        chains and trimmed to ``n_samples``. All non-fixed sites are updated,
        so sites that are neither fixed nor of interest are marginalised.

        Returns an array (n_samples, V). ``self.acceptance_rate`` and
        ``self.last_state`` (n_chains, V) describe the finished run.
        """
        if n_samples < 1 or thin < 1 or burn < 0:
            raise ValueError("need n_samples >= 1, thin >= 1, burn >= 0")
        fixed = {int(k): float(v) for k, v in (fixed or {}).items()}
        for k in fixed:
            if not 0 <= k < self.V:
                raise IndexError(f"fixed site {k} outside [0, {self.V})")
        free = np.array([i for i in range(self.V) if i not in fixed], dtype=int)
        if free.size == 0:
            raise ValueError("every site is fixed; nothing to sample")
        phi = self._init(init, fixed)

        C = self.n_chains
        n_rec = -(-n_samples // C)
        out = np.empty((n_rec, C, self.V))
        self._acc = self._prop = 0
        for _ in range(burn):
            self._sweep(phi, free)
        for s in range(n_rec):
            for _ in range(thin):
                self._sweep(phi, free)
            out[s] = phi
        self.acceptance_rate = self._acc / max(self._prop, 1)
        self.last_state = phi.copy()
        return out.reshape(n_rec * C, self.V)[:n_samples]

    def sample_conditional(self, fixed: dict, n_samples: int = 1000, burn: int = 500,
                           thin: int = 1, init=None) -> np.ndarray:
        """Shorthand for ``sample(..., fixed=fixed)``."""
        return self.sample(n_samples=n_samples, burn=burn, thin=thin, init=init, fixed=fixed)


class MetropolisSampler(_BaseSampler):
    """Single-site Metropolis with independent uniform proposals on
    [-r/2, r/2] (the paper's scheme), vectorised over ``n_chains`` chains.

    With one chain a scalar loop is used (numpy call overhead dominates
    there); with several chains each site update is one vectorised step.
    """

    def _sweep(self, phi, free):
        C, rng, half = phi.shape[0], self.rng, self.half
        n = len(free)
        props = rng.uniform(-half, half, size=(n, C))
        log_u = np.log(rng.random((n, C)))
        W2, a, mu, lam = 2.0 * self.W, self.a, self.mu, self.lam
        acc_total = 0
        if C == 1:
            x = phi[0]
            pr, lu = props[:, 0].tolist(), log_u[:, 0].tolist()
            for k, i in enumerate(free.tolist()):
                new, old = pr[k], x[i]
                h = a[i] + W2[i] @ x
                d, s = new - old, new + old
                dS = d * (-h + mu[i] * s + lam[i] * s * (new * new + old * old))
                if lu[k] < -dS:
                    x[i] = new
                    acc_total += 1
        else:
            for k, i in enumerate(free):
                new, old = props[k], phi[:, i]
                h = a[i] + phi @ W2[i]
                d, s = new - old, new + old
                dS = d * (-h + mu[i] * s + lam[i] * s * (new * new + old * old))
                acc = log_u[k] < -dS
                acc_total += np.count_nonzero(acc)
                phi[:, i] = np.where(acc, new, old)
        self._acc += int(acc_total)
        self._prop += n * C


class HeatBathSampler(_BaseSampler):
    """Heat-bath (Gibbs) sampler: each site is redrawn from its exact 1-D
    conditional p(phi_i | phi_-i) ∝ exp(h_i phi - mu_i phi^2 - lam_i phi^4),
    tabulated on ``n_grid`` cells of [-r/2, r/2] (inverse CDF, then uniform
    inside the cell). Every update is accepted, so it mixes faster than
    Metropolis with a uniform proposal, at the cost of O(n_grid) per update.
    """

    def __init__(self, *args, n_grid: int = 256, **kw):
        super().__init__(*args, **kw)
        if n_grid < 16:
            raise ValueError("n_grid must be >= 16")
        self.dx = self.proposal_range / n_grid
        self.grid = -self.half + self.dx * (np.arange(n_grid) + 0.5)
        self._g2, self._g4 = self.grid**2, self.grid**4

    def _sweep(self, phi, free):
        C, rng, g = phi.shape[0], self.rng, self.grid
        u = rng.random((len(free), C, 1))
        jitter = (rng.random((len(free), C)) - 0.5) * self.dx
        W, a, mu, lam = self.W, self.a, self.mu, self.lam
        for k, i in enumerate(free):
            h = a[i] + 2.0 * (phi @ W[:, i])
            logp = h[:, None] * g - mu[i] * self._g2 - lam[i] * self._g4
            p = np.exp(logp - logp.max(axis=1, keepdims=True))
            cdf = np.cumsum(p, axis=1)
            idx = (cdf < u[k] * cdf[:, -1:]).sum(axis=1)
            phi[:, i] = g[np.minimum(idx, g.size - 1)] + jitter[k]
        self._acc += len(free) * C
        self._prop += len(free) * C


SAMPLERS = {"metropolis": MetropolisSampler, "heatbath": HeatBathSampler}
