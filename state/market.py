"""
5D market state per asset from §I.3 of the mathematical framework.
s_t = (p_t, v_t, ℓ_t, κ_t, ι_t) ∈ ℝ⁵

For n assets: S_t ∈ ℝ^{n×5}  (or H = L²([0,1]×[0,T], ℝ⁵) in the continuum limit)
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Optional


@dataclass
class AssetState:
    ticker: str
    p_t: float          # log price
    v_t: float          # log volume
    l_t: float          # leverage ratio (capital structure)
    kappa_t: float      # shares outstanding (equity size)
    iota_t: float       # information disclosure level ∈ [0, 1]

    def to_vec(self) -> np.ndarray:
        return np.array([self.p_t, self.v_t, self.l_t, self.kappa_t, self.iota_t])


@dataclass
class MarketState:
    timestamp: float
    assets: dict[str, AssetState] = field(default_factory=dict)
    macro: dict = field(default_factory=dict)   # DFF, CPI, VIX, T10Y2Y, ...

    def to_matrix(self) -> np.ndarray:
        """Return S_t ∈ ℝ^{n×5}."""
        return np.stack([s.to_vec() for s in self.assets.values()])

    def prices(self) -> dict[str, float]:
        return {t: np.exp(s.p_t) for t, s in self.assets.items()}

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, timestamp: float) -> "MarketState":
        """Build MarketState from a row of the market_features table."""
        assets = {}
        for ticker, row in df.iterrows():
            assets[ticker] = AssetState(
                ticker=ticker,
                p_t=np.log(row.get("close", 1.0) + 1e-8),
                v_t=np.log(row.get("volume", 1.0) + 1.0),
                l_t=row.get("leverage", 1.0),
                kappa_t=row.get("shares_outstanding", 1e9),
                iota_t=row.get("disclosure_level", 0.5),
            )
        return cls(timestamp=timestamp, assets=assets)
