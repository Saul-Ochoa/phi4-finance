
import yfinance as yf
import numpy as np
import pandas as pd

def load_returns(tickers, period="5y", interval="1d", log=True, normalize="01"):
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]] if "Close" in data else data
    close = close.dropna()
    if log:
        rets = np.log(close / close.shift(1)).dropna()
    else:
        rets = close.pct_change().dropna()
    if normalize == "01":
        min_ = rets.min()
        max_ = rets.max()
        rets_norm = 2*(rets - min_)/(max_-min_) - 1
    elif normalize == "absmax":
        rets_norm = rets / rets.abs().max()
    else:
        rets_norm = rets
    return rets_norm
