"""Price loading from Yahoo Finance (optional dependency ``yfinance``).

Returns are left in return units; scale them with ``phi4finance.Scaler``
fitted on the training window only.

Notes on reproducing Bachtis et al. (2026): the paper uses WRDS data.
Yahoo's ``adjusted=True`` close is split- and dividend-adjusted; with
``adjusted=False`` it is split-adjusted only. Yahoo does not provide fully
unadjusted prices, so the paper's unadjusted log-returns (Appendix A.1) can
only be approximated.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _download(tickers, start, end, period, interval, adjusted):
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("loading prices needs yfinance: pip install 'phi4-finance[data]'") from e
    kw = {"start": start, "end": end} if (start or end) else {"period": period}
    data = yf.download(list(tickers), interval=interval, auto_adjust=adjusted,
                       progress=False, **kw)
    if data is None or len(data) == 0:
        raise RuntimeError(f"no data returned for {list(tickers)}")
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    if list(close.columns) == ["Close"]:
        close.columns = list(tickers)
    return close


def load_prices(tickers, start=None, end=None, period: str = "5y", interval: str = "1d",
                adjusted: bool = True, cache_dir=None, min_coverage: float = 0.95) -> pd.DataFrame:
    """Daily closes, one column per ticker, rows with any missing price dropped.

    Parameters
    ----------
    start, end : dates (``end`` exclusive, as in yfinance). If both are None,
        ``period`` is used.
    adjusted : split- and dividend-adjusted closes (True) or split-adjusted
        only (False).
    cache_dir : folder for a CSV cache; only used with explicit ``start`` and
        ``end`` so a cached file never goes stale.
    min_coverage : tickers with prices on fewer than this share of dates are
        dropped with a warning (e.g. companies listed after ``start``).
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    path = None
    if cache_dir is not None and start is not None and end is not None:
        key = hashlib.md5(",".join(tickers).encode()).hexdigest()[:10]
        path = Path(cache_dir) / f"prices_{key}_{start}_{end}_{'adj' if adjusted else 'split'}_{interval}.csv"
        if path.exists():
            close = pd.read_csv(path, index_col=0, parse_dates=True)
            return close[[t for t in tickers if t in close.columns]]
    close = _download(tickers, start, end, period, interval, adjusted)
    close = close[[t for t in tickers if t in close.columns]]
    coverage = close.notna().mean()
    dropped = coverage[coverage < min_coverage].index.tolist()
    missing = [t for t in tickers if t not in close.columns]
    if dropped or missing:
        warnings.warn(f"dropped tickers with insufficient data: {missing + dropped}")
    close = close.drop(columns=dropped).dropna()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        close.to_csv(path)
    return close


def load_returns(tickers, period: str = "5y", interval: str = "1d", log: bool = True,
                 start=None, end=None, adjusted: bool = True, cache_dir=None,
                 min_coverage: float = 0.95) -> pd.DataFrame:
    """Daily log-returns (``log=True``) or simple returns of the closes from
    ``load_prices``; see that function for the other parameters."""
    close = load_prices(tickers, start=start, end=end, period=period, interval=interval,
                        adjusted=adjusted, cache_dir=cache_dir, min_coverage=min_coverage)
    rets = np.log(close / close.shift(1)) if log else close.pct_change()
    return rets.dropna()
