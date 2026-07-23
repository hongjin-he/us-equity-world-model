"""
Information set for each agent — what they observe at each timestep.
Asymmetric information is the central mechanism of this model.
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class InformationSet:
    agent_id: str

    # Public market data (all agents observe this)
    ohlcv: Optional[pd.DataFrame] = None          # price/volume history
    macro_indicators: dict = field(default_factory=dict)  # CPI, rates, VIX, etc.
    public_filings: list = field(default_factory=list)    # 8-K, earnings releases

    # Semi-public (available but requires effort to process)
    sec_13f_snapshot: Optional[pd.DataFrame] = None  # quarterly institutional holdings
    options_flow: Optional[pd.DataFrame] = None

    # Private signals (agent-specific alpha)
    proprietary_signals: dict = field(default_factory=dict)

    # Information lag: some agents process data faster than others
    lag_seconds: float = 0.0

    def observe(self, market_state: "MarketState", timestamp) -> "InformationSet":
        """Filter global market state through this agent's information set."""
        raise NotImplementedError
