"""Metropolis sampler for the phi^4 lattice action (eq. 1).

    S(phi) = - sum_{i,j} w_ij phi_i phi_j + sum_i mu_i phi_i^2
             + sum_i lam_i phi_i^4 - sum_i a_i phi_i

Proposals are drawn uniformly on [-r/2, r/2] with r = ``proposal_range``
(r = 3 gives the paper's [-1.5, 1.5], Appendix A.1). Because the proposal is
independent of the current value and bounded, the chain samples exp(-S)
*truncated* to that hypercube, not the theory on the whole real line.
"""
from __future__ import annotations

import numpy as np


class MetropolisSampler:
    """Single-site Metropolis sampler.

    One *sweep* proposes a new value once for every free site. The change in
    the action for site i is computed locally in O(V):

        dS = -h_i (new - old) + mu_i (new^2 - old^2) + lam_i (new^4 - old^4),
        h_i = a_i + 2 sum_j w_ij phi_j        (W symmetric, zero diagonal)
    """

    def __init__(self, W, mu, lam, a, proposal_range: float = 3.0, rng=None):
        a = np.asarray(a, dtype=float)
        V = a.shape[0]
        W = np.array(W, dtype=float)
        if W.shape != (V, V):
            raise ValueError(f"W must be ({V}, {V}), got {W.shape}")
        if not np.allclose(W, W.T):
            raise ValueError("W must be symmetric")
        np.fill_diagonal(W, 0.0)
        self.W = W
        self.mu = np.broadcast_to(np.asarray(mu, dtype=float), (V,)).copy()
        self.lam = np.broadcast_to(np.asarray(lam, dtype=float), (V,)).copy()
        self.a = a
        self.V = V
        self.proposal_range = float(proposal_range)
        self.half = self.proposal_range / 2.0
        self.rng = rng if rng is not None else np.random.default_rng()

    # ----------------------------------------------------------- action
    def action(self, phi) -> float:
        phi = np.asarray(phi, dtype=float)
        return float(-phi @ self.W @ phi + self.mu @ phi**2 + self.lam @ phi**4 - self.a @ phi)

    def local_field(self, phi, i: int) -> float:
        return self.a[i] + 2.0 * self.W[i] @ phi

    def delta_action(self, phi, i: int, new: float) -> float:
        old = phi[i]
        h = self.local_field(phi, i)
        return (-h * (new - old) + self.mu[i] * (new**2 - old**2)
                + self.lam[i] * (new**4 - old**4))

    # ----------------------------------------------------------- sweeps
    def _sweep(self, phi, free) -> int:
        rng, half = self.rng, self.half
        proposals = rng.uniform(-half, half, size=len(free))
        log_u = np.log(rng.random(len(free)))
        accepted = 0
        for k, i in enumerate(free):
            dS = self.delta_action(phi, i, proposals[k])
            if dS <= 0.0 or log_u[k] < -dS:
                phi[i] = proposals[k]
                accepted += 1
        return accepted

    def sample(self, n_samples: int = 1000, burn: int = 500, thin: int = 1,
               init=None, fixed: dict | None = None) -> np.ndarray:
        """Draw configurations from p(phi) or, with ``fixed``, from
        p(phi_free | phi_fixed).

        Parameters
        ----------
        n_samples : number of configurations returned.
        burn : sweeps discarded before the first sample.
        thin : sweeps between consecutive samples (>= 1).
        init : optional starting configuration (length V).
        fixed : optional {site: value} clamped for the whole chain. Every
            other site is updated, so non-target free sites are marginalised
            rather than left at their initial values.

        Returns
        -------
        array (n_samples, V) of full configurations. The acceptance rate of
        the last call is stored in ``self.acceptance_rate``.
        """
        if n_samples < 1 or thin < 1 or burn < 0:
            raise ValueError("need n_samples >= 1, thin >= 1, burn >= 0")
        fixed = dict(fixed or {})
        for k in fixed:
            if not 0 <= k < self.V:
                raise IndexError(f"fixed site {k} outside [0, {self.V})")
        if init is None:
            phi = self.rng.uniform(-min(1.0, self.half), min(1.0, self.half), size=self.V)
        else:
            phi = np.array(init, dtype=float)
            if phi.shape != (self.V,):
                raise ValueError(f"init must have shape ({self.V},)")
        for k, v in fixed.items():
            phi[k] = float(v)
        free = np.array([i for i in range(self.V) if i not in fixed], dtype=int)
        if free.size == 0:
            raise ValueError("every site is fixed; nothing to sample")

        out = np.empty((n_samples, self.V))
        accepted = proposed = 0
        for _ in range(burn):
            accepted += self._sweep(phi, free); proposed += free.size
        for s in range(n_samples):
            for _ in range(thin):
                accepted += self._sweep(phi, free); proposed += free.size
            out[s] = phi
        self.acceptance_rate = accepted / max(proposed, 1)
        self.last_state = phi.copy()
        return out

    def sample_conditional(self, fixed: dict, n_samples: int = 1000, burn: int = 500,
                           thin: int = 1, init=None) -> np.ndarray:
        """Shorthand for ``sample(..., fixed=fixed)``."""
        return self.sample(n_samples=n_samples, burn=burn, thin=thin, init=init, fixed=fixed)
