"""Exact one-site conditionals (no MCMC).

With every site but one known, the phi^4 conditional is one-dimensional:

    p(phi_t | phi_-t) ∝ exp(h_t phi_t - mu_t phi_t^2 - lam_t phi_t^4),
    h_t = a_t + 2 sum_j w_tj phi_j

on the truncated support [-r/2, r/2]. Its normaliser is a 1-D integral, done
here by the midpoint rule on a fine grid. This covers the paper's forecasting
question p(phi_1 | phi_0, ..., phi_-148) (eq. 11) exactly.
"""
from __future__ import annotations

import numpy as np


class ConditionalDistribution:
    """A 1-D distribution tabulated on equally spaced cell midpoints.

    ``grid`` holds the cell centres and ``prob`` the probability of each
    cell (sums to 1). Values are in whatever units ``grid`` is in; use
    ``affine`` (or ``Scaler``-aware helpers in ``Phi4Model``) to change units.
    """

    def __init__(self, grid, prob):
        self.grid = np.asarray(grid, dtype=float)
        p = np.asarray(prob, dtype=float)
        self.prob = p / p.sum()
        self.dx = float(abs(self.grid[1] - self.grid[0]))

    # ---------------------------------------------------------------- moments
    def mean(self) -> float:
        return float(self.prob @ self.grid)

    def var(self) -> float:
        m = self.mean()
        # + dx^2/12: variance of the uniform spread inside each cell
        return float(self.prob @ (self.grid - m) ** 2 + self.dx**2 / 12.0)

    def std(self) -> float:
        return float(np.sqrt(self.var()))

    def moment(self, k: int) -> float:
        return float(self.prob @ self.grid**k)

    # ---------------------------------------------------------------- shape
    def pdf(self):
        """Density values at the grid points (integrates to 1)."""
        return self.prob / self.dx

    def cdf(self, x):
        """P(X <= x), linear inside each cell."""
        edges = np.concatenate([[self.grid[0] - self.dx / 2], self.grid + self.dx / 2])
        cum = np.concatenate([[0.0], np.cumsum(self.prob)])
        if edges[0] > edges[-1]:  # decreasing grid after a negative affine map
            edges, cum = edges[::-1], 1.0 - cum[::-1]
        return np.interp(x, edges, cum)

    def quantile(self, q):
        edges = np.concatenate([[self.grid[0] - self.dx / 2], self.grid + self.dx / 2])
        cum = np.concatenate([[0.0], np.cumsum(self.prob)])
        if edges[0] > edges[-1]:
            edges, cum = edges[::-1], 1.0 - cum[::-1]
        out = np.interp(q, cum, edges)
        return float(out) if np.ndim(out) == 0 else out

    def interval(self, level: float = 0.9):
        """Central interval with probability ``level``."""
        a = (1.0 - level) / 2.0
        return self.quantile(a), self.quantile(1.0 - a)

    def sample(self, n: int, rng=None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        idx = rng.choice(self.grid.size, size=n, p=self.prob)
        return self.grid[idx] + (rng.random(n) - 0.5) * self.dx

    # ---------------------------------------------------------------- scoring
    def logpdf(self, y) -> float:
        """Log density at y (piecewise constant per cell; -inf outside the grid)."""
        lo = min(self.grid[0], self.grid[-1]) - self.dx / 2
        k = np.floor((y - lo) / self.dx).astype(int) if np.ndim(y) else int(np.floor((y - lo) / self.dx))
        order = np.argsort(self.grid)
        p = self.prob[order] / self.dx
        ok = (k >= 0) & (k < p.size)
        out = np.where(ok, np.log(np.maximum(p[np.clip(k, 0, p.size - 1)], 1e-300)), -np.inf)
        return float(out) if np.ndim(out) == 0 else out

    def pit(self, y):
        """Probability integral transform F(y); uniform on [0, 1] if the forecasts are calibrated."""
        return self.cdf(y)

    def crps(self, y) -> float:
        """Continuous ranked probability score E|X - y| - E|X - X'| / 2 (lower is better),
        treating each cell as a point mass at its centre."""
        order = np.argsort(self.grid)
        g, p = self.grid[order], self.prob[order]
        C = np.cumsum(p)
        e_xx = 2.0 * np.sum(p * g * (2.0 * C - p - 1.0))      # E|X - X'| for sorted point masses
        return float(p @ np.abs(g - y) - 0.5 * e_xx)

    @classmethod
    def gaussian(cls, mean: float, std: float, n_grid: int = 801, width: float = 8.0):
        """A normal distribution tabulated on +- ``width`` standard deviations."""
        g = mean + std * np.linspace(-width, width, n_grid)
        return cls(g, np.exp(-0.5 * ((g - mean) / std) ** 2))

    # ---------------------------------------------------------------- units
    def affine(self, scale, shift=0.0) -> "ConditionalDistribution":
        """Distribution of ``scale * X + shift``."""
        return ConditionalDistribution(scale * self.grid + shift, self.prob)

    def __repr__(self):
        return f"ConditionalDistribution(mean={self.mean():.4g}, std={self.std():.4g})"


def site_grid(proposal_range: float = 3.0, n_grid: int = 801) -> np.ndarray:
    """Cell midpoints covering [-r/2, r/2]."""
    dx = proposal_range / n_grid
    return -proposal_range / 2.0 + dx * (np.arange(n_grid) + 0.5)


def exact_conditional(W, mu, lam, a, phi, target: int, proposal_range: float = 3.0,
                      n_grid: int = 801) -> ConditionalDistribution:
    """p(phi_target | all other sites) for a full configuration ``phi`` (V,)
    whose ``target`` entry is ignored."""
    W = np.asarray(W, dtype=float)
    phi = np.array(phi, dtype=float)
    phi[target] = 0.0
    V = len(phi)
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (V,))
    lam = np.broadcast_to(np.asarray(lam, dtype=float), (V,))
    h = a[target] + 2.0 * (W[target] @ phi)
    g = site_grid(proposal_range, n_grid)
    logp = h * g - mu[target] * g**2 - lam[target] * g**4
    return ConditionalDistribution(g, np.exp(logp - logp.max()))
