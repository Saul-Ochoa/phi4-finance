"""phi4-finance: phi^4 quantum field theory for financial time series.

Implementation of Bachtis, Berman & Schelpe, "Modeling financial time series
with phi^4 quantum field theory", Physica A 701 (2026) 132033,
doi:10.1016/j.physa.2026.132033.
"""
from .data import load_returns
from .model import Phi4Model
from .preprocessing import Scaler, lag_embed
from .sampler import MetropolisSampler

__all__ = ["Phi4Model", "MetropolisSampler", "Scaler", "lag_embed", "load_returns"]
__version__ = "0.2.0"
