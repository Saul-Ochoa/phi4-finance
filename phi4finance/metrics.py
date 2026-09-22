
import numpy as np
from scipy.stats import kurtosis

def market_mean(returns):
    return returns.mean(axis=1)

def market_kurtosis(returns):
    return np.array([kurtosis(returns[i,:], fisher=False) for i in range(returns.shape[0])])

def sma(x, window=250):
    import pandas as pd
    return pd.Series(x).rolling(window).mean().values
