"""Data scaling and lag embedding (Bachtis et al. 2026, Appendix A.1).

The phi^4 model is sampled on a bounded support, so returns must be mapped to
roughly [-1, 1] before training. The mapping has to be:

* fitted on the training window only (no look-ahead into the forecast days);
* stored, so conditioning values and samples can move between return units
  and model units (eqs. 13 and 15 of the paper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_METHODS = ("minmax", "absmax", "none")


class Scaler:
    """Invertible column-wise scaler.

    Parameters
    ----------
    method : {"minmax", "absmax", "none"}
        ``"minmax"`` maps [min, max] to [-1, 1] (eq. 14). It moves a zero
        return away from 0, which the biases a_i must then absorb.
        ``"absmax"`` divides by max|X| (eq. 12); it keeps 0 at 0 and the sign
        of every return, so it is usually the better choice for returns.
        ``"none"`` is the identity.

    Fitted on a 1-D series the scaler stores scalar parameters, which is what
    the lag-embedded forecasting set-up needs (every site is the same stock).
    Fitted on a 2-D array or DataFrame it stores one set per column.
    """

    def __init__(self, method: str = "minmax"):
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        self.method = method
        self.min_ = self.max_ = self.absmax_ = None
        self.columns_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, X) -> "Scaler":
        if isinstance(X, pd.DataFrame):
            self.columns_ = list(X.columns)
        arr = np.asarray(X, dtype=float)
        if arr.ndim not in (1, 2):
            raise ValueError("X must be 1-D or 2-D")
        axis = None if arr.ndim == 1 else 0
        self.min_ = np.nanmin(arr, axis=axis)
        self.max_ = np.nanmax(arr, axis=axis)
        self.absmax_ = np.nanmax(np.abs(arr), axis=axis)
        if np.any(self.max_ - self.min_ == 0) or np.any(self.absmax_ == 0):
            raise ValueError("a column is constant; it cannot be scaled")
        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # ------------------------------------------------------------ helpers
    def _check(self):
        if self.min_ is None:
            raise RuntimeError("Scaler is not fitted; call fit() on the training data first")

    def _p(self, value, cols):
        value = np.asarray(value)
        if value.ndim == 0 or cols is None:
            return value
        return value[cols]

    @staticmethod
    def _wrap(out, like):
        if isinstance(like, pd.DataFrame):
            return pd.DataFrame(out, index=like.index, columns=like.columns)
        if isinstance(like, pd.Series):
            return pd.Series(out, index=like.index, name=like.name)
        if np.ndim(out) == 0:
            return float(out)
        return out

    # ------------------------------------------------------------ transforms
    def transform(self, X, cols=None):
        """Return units -> model units. ``cols`` selects which fitted columns
        the last axis of ``X`` corresponds to (int or list of ints)."""
        self._check()
        arr = np.asarray(X, dtype=float)
        if self.method == "minmax":
            lo, hi = self._p(self.min_, cols), self._p(self.max_, cols)
            out = 2.0 * (arr - lo) / (hi - lo) - 1.0
        elif self.method == "absmax":
            out = arr / self._p(self.absmax_, cols)
        else:
            out = arr.copy()
        return self._wrap(out, X)

    def inverse_transform(self, X, cols=None):
        """Model units -> return units (eqs. 13 and 15)."""
        self._check()
        arr = np.asarray(X, dtype=float)
        if self.method == "minmax":
            lo, hi = self._p(self.min_, cols), self._p(self.max_, cols)
            out = (arr + 1.0) * (hi - lo) / 2.0 + lo
        elif self.method == "absmax":
            out = arr * self._p(self.absmax_, cols)
        else:
            out = arr.copy()
        return self._wrap(out, X)

    def scale_factor(self, cols=None):
        """Multiplier that converts a spread (std, MAE) from model units to
        return units."""
        self._check()
        if self.method == "minmax":
            return (self._p(self.max_, cols) - self._p(self.min_, cols)) / 2.0
        if self.method == "absmax":
            return self._p(self.absmax_, cols)
        return 1.0

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        self._check()
        conv = lambda v: np.asarray(v).tolist()
        return {"method": self.method, "min": conv(self.min_), "max": conv(self.max_),
                "absmax": conv(self.absmax_), "columns": self.columns_}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        s = cls(d["method"])
        s.min_, s.max_, s.absmax_ = (np.asarray(d[k], dtype=float) for k in ("min", "max", "absmax"))
        s.columns_ = d.get("columns")
        return s

    def __repr__(self):
        return f"Scaler(method={self.method!r}, fitted={self.min_ is not None})"


def lag_embed(series, window: int) -> np.ndarray:
    """Stack overlapping windows of a 1-D series (Section 3.5).

    Row ``t`` is ``[x_t, x_{t+1}, ..., x_{t+window-1}]`` in chronological
    order, so the last column is the most recent day. With ``window=150`` each
    row is one training configuration of the paper's forecasting model: the
    first 149 sites are known history and site 149 is the day to forecast.
    """
    x = np.asarray(series, dtype=float).ravel()
    if window < 2:
        raise ValueError("window must be >= 2")
    if len(x) < window:
        raise ValueError(f"series has {len(x)} points, fewer than window={window}")
    return np.lib.stride_tricks.sliding_window_view(x, window).copy()
