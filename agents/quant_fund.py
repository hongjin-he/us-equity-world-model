"""
Quant fund agent. Strategy style is parameterized at construction time,
inferred from public signals (team composition, 13F turnover, factor exposures).
"""
from dataclasses import dataclass
from enum import Enum
from typing import List

from agents.base import BaseAgent, Order
from state.portfolio import Portfolio
from state.information import InformationSet


class StrategyStyle(Enum):
    MOMENTUM = "momentum"
    STAT_ARB = "stat_arb"
    ML_ALPHA = "ml_alpha"        # deep learning signal-based
    LOW_VOL = "low_vol"
    MACRO_QUANT = "macro_quant"


@dataclass
class QuantFundProfile:
    """
    Inferred from public information:
    - 13F filings: AUM, turnover, concentration
    - Team composition: academics (stat_arb) vs ML engineers (ml_alpha)
    - Factor exposure regressions on public holdings history
    """
    name: str
    aum_usd: float
    strategy_style: StrategyStyle
    avg_holding_period_days: float   # estimated from 13F turnover
    factor_loadings: dict            # {factor_name: loading} estimated from returns
    leverage_estimate: float = 1.0
    rebalance_freq_days: int = 63    # quarterly default


class QuantFundAgent(BaseAgent):
    def __init__(self, profile: QuantFundProfile, portfolio: Portfolio, info_set: InformationSet):
        super().__init__(agent_id=profile.name, portfolio=portfolio, info_set=info_set)
        self.profile = profile

    def act(self, timestamp: float) -> List[Order]:
        # Strategy dispatch — each style will have its own signal generation logic
        if self.profile.strategy_style == StrategyStyle.MOMENTUM:
            return self._momentum_act(timestamp)
        elif self.profile.strategy_style == StrategyStyle.STAT_ARB:
            return self._stat_arb_act(timestamp)
        elif self.profile.strategy_style == StrategyStyle.ML_ALPHA:
            return self._ml_alpha_act(timestamp)
        return []

    def _momentum_act(self, timestamp: float) -> List[Order]:
        raise NotImplementedError

    def _stat_arb_act(self, timestamp: float) -> List[Order]:
        raise NotImplementedError

    def _ml_alpha_act(self, timestamp: float) -> List[Order]:
        raise NotImplementedError
