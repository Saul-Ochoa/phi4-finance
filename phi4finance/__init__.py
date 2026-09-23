"""phi4-finance: phi^4 quantum field theory for financial time series.

Implementation of Bachtis, Berman & Schelpe, "Modeling financial time series
with phi^4 quantum field theory", Physica A 701 (2026) 132033,
doi:10.1016/j.physa.2026.132033.
"""
from .data import load_prices, load_returns
from .inference import ConditionalDistribution
from .model import Phi4Model
from .preprocessing import Scaler, lag_embed
from .rolling import RollingPhi4
from .sampler import HeatBathSampler, MetropolisSampler
from .structure import Tying, lag_embed_panel
from .volatility import EWMAVol, GARCHVol, devolatilize

__all__ = ["Phi4Model", "MetropolisSampler", "HeatBathSampler", "ConditionalDistribution",
           "RollingPhi4", "Scaler", "lag_embed", "lag_embed_panel", "Tying", "EWMAVol", "GARCHVol",
           "devolatilize", "load_prices", "load_returns"]
__version__ = "0.5.1"
