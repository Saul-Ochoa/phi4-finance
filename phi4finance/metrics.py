"""Market statistics used in Section 3.1 (Fig. 1) and Appendix A.3."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


def market_mean(returns) -> np.ndarray:
    """Cross-sectional mean per day, (T, V) -> (T,). Equals the magnetisation m (eq. 19-20)."""
    return np.asarray(returns, dtype=float).mean(axis=1)


def market_kurtosis(returns) -> np.ndarray:
    """Cross-sectional (Pearson, non-excess) kurtosis per day, (T, V) -> (T,)."""
    return kurtosis(np.asarray(returns, dtype=float), axis=1, fisher=False)


def binarize(returns) -> np.ndarray:
    """Sign of returns, the input an Ising model would see."""
    return np.sign(np.asarray(returns, dtype=float))


def sma(x, window: int = 250) -> np.ndarray:
    """Simple moving average over the ``window`` previous values including t (eq. 18)."""
    return pd.Series(np.asarray(x, dtype=float)).rolling(window).mean().to_numpy()


def magnetization(samples) -> float:
    """<m> over configurations (n, V)."""
    return float(market_mean(samples).mean())


def susceptibility(samples) -> float:
    """chi_m = <m^2> - <m>^2 over configurations (n, V)."""
    return float(market_mean(samples).var())


# ---------------------------------------------------------------- Section 3.2
def sign_product_matrix(day_returns) -> np.ndarray:
    """rho_ij = sgn(r_i) sgn(r_j) for one day's returns (Section 3.2)."""
    s = np.sign(np.asarray(day_returns, dtype=float))
    return np.outer(s, s)


def sign_agreement(A, B) -> float:
    """Share of pairs i < j where sgn(A_ij) == sgn(B_ij)."""
    A, B = np.asarray(A), np.asarray(B)
    iu = np.triu_indices(A.shape[0], 1)
    return float(np.mean(np.sign(A[iu]) == np.sign(B[iu])))


# ---------------------------------------------------------------- forecasts
def mae(pred, truth) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(truth, float))))


def mae_se(pred, truth) -> float:
    """Standard error of the MAE (std of absolute errors / sqrt(n))."""
    e = np.abs(np.asarray(pred, float) - np.asarray(truth, float))
    return float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else float("nan")


def coverage(truth, lower, upper) -> float:
    """Share of outcomes inside [lower, upper]."""
    y = np.asarray(truth, float)
    return float(np.mean((y >= np.asarray(lower, float)) & (y <= np.asarray(upper, float))))


def hit_rate(pred, truth) -> float:
    """Share of days where the forecast has the sign of the outcome."""
    return float(np.mean(np.sign(np.asarray(pred, float)) == np.sign(np.asarray(truth, float))))
