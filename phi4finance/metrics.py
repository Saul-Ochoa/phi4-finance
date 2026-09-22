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
