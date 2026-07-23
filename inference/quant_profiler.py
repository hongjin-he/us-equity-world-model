"""
Infer quant fund strategy style from public signals.

Signal sources (all free/public):
1. 13F turnover rate → holding period estimate
2. Factor regression on quarterly holdings changes → factor loadings
3. Team composition from public bios → technology stack inference
4. AUM concentration (top-10 holdings %) → breadth of strategy
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from agents.quant_fund import StrategyStyle, QuantFundProfile


@dataclass
class PublicSignals:
    name: str
    aum_usd: float
    quarterly_turnover_pct: float       # from 13F QoQ holding changes
    top10_concentration_pct: float      # % of portfolio in top-10 names
    team_has_ml_engineers: bool         # inferred from LinkedIn/public bios
    team_has_phd_stat_math: bool
    known_factor_exposures: dict        # {factor: t-stat} from regression


def infer_strategy_style(signals: PublicSignals) -> StrategyStyle:
    if signals.quarterly_turnover_pct > 80 and signals.team_has_ml_engineers:
        return StrategyStyle.ML_ALPHA
    if signals.quarterly_turnover_pct > 60 and signals.top10_concentration_pct < 20:
        return StrategyStyle.STAT_ARB
    if signals.known_factor_exposures.get("momentum", 0) > 2.0:
        return StrategyStyle.MOMENTUM
    if signals.known_factor_exposures.get("low_vol", 0) > 2.0:
        return StrategyStyle.LOW_VOL
    return StrategyStyle.MOMENTUM  # default fallback


def build_profile(signals: PublicSignals) -> QuantFundProfile:
    style = infer_strategy_style(signals)
    avg_holding = max(5, 63 * (1 - signals.quarterly_turnover_pct / 100))

    return QuantFundProfile(
        name=signals.name,
        aum_usd=signals.aum_usd,
        strategy_style=style,
        avg_holding_period_days=avg_holding,
        factor_loadings=signals.known_factor_exposures,
    )
