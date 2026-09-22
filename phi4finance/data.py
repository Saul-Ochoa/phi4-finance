"""Price loading. Returns are left in return units; scale them with
``phi4finance.Scaler`` fitted on the training window only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_returns(tickers, period: str = "5y", interval: str = "1d", log: bool = True,
                 start=None, end=None) -> pd.DataFrame:
    """Daily (log-)returns of adjusted closes from Yahoo Finance.

    Requires the optional dependency ``yfinance`` (``pip install phi4-finance[data]``).
    Rows with a missing price for any ticker are dropped.
    """
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("load_returns needs yfinance: pip install 'phi4-finance[data]'") from e
    if isinstance(tickers, str):
        tickers = [tickers]
    kw = {"start": start, "end": end} if (start or end) else {"period": period}
    data = yf.download(list(tickers), interval=interval, auto_adjust=True, progress=False, **kw)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    if list(close.columns) == ["Close"]:
        close.columns = list(tickers)
    close = close[list(tickers)].dropna()
    rets = np.log(close / close.shift(1)) if log else close.pct_change()
    return rets.dropna()
