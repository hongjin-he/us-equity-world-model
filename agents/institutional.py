"""
Institutional agent types at L2/L3 of the hierarchy (§VI).

Each agent class has a distinct objective function, information set, and
capacity constraint. The resulting heterogeneity is what sustains the
MFG equilibrium — a market of identical agents would trivially collapse.

Agent taxonomy (6 classes following the Avatar analogy):
  HFT        : speed → adverse selection edge
  StatArb    : cross-sectional mean reversion; capacity-limited
  TrendFollower: momentum strategies; herding risk
  MarketMaker: bid-ask capture; inventory risk management
  FundamentalLong: DCF valuation; slow signal, large AUM
  CrisisHedge: tail protection; convex payoffs

Reference: Alpha Flow §VI; Kyle (1985) for information-based trading.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class AgentType(Enum):
    HFT              = auto()
    STAT_ARB         = auto()
    TREND_FOLLOWER   = auto()
    MARKET_MAKER     = auto()
    FUNDAMENTAL_LONG = auto()
    CRISIS_HEDGE     = auto()


@dataclass
class AgentProfile:
    """Static parameters that define an agent class."""
    aum_bn: float          # typical AUM in $B
    holding_period_days: float
    leverage_limit: float  # max gross leverage
    capacity_signal_decay: float  # how fast alpha decays with $ invested
    crowding_aversion: float      # λ in crowding penalty (higher = avoids crowded trades)
    information_latency_ms: float # typical data latency in milliseconds


PROFILES: dict[AgentType, AgentProfile] = {
    AgentType.HFT: AgentProfile(
        aum_bn=5.0,
        holding_period_days=1 / 390,  # intraday ~1 bar
        leverage_limit=20.0,
        capacity_signal_decay=0.9,    # very fast decay with size
        crowding_aversion=0.1,
        information_latency_ms=0.01,  # microsecond infrastructure
    ),
    AgentType.STAT_ARB: AgentProfile(
        aum_bn=20.0,
        holding_period_days=5.0,
        leverage_limit=8.0,
        capacity_signal_decay=0.6,
        crowding_aversion=0.5,
        information_latency_ms=10.0,
    ),
    AgentType.TREND_FOLLOWER: AgentProfile(
        aum_bn=50.0,
        holding_period_days=63.0,
        leverage_limit=4.0,
        capacity_signal_decay=0.3,
        crowding_aversion=0.3,  # creates herding → momentum crashes
        information_latency_ms=100.0,
    ),
    AgentType.MARKET_MAKER: AgentProfile(
        aum_bn=10.0,
        holding_period_days=0.5,
        leverage_limit=15.0,
        capacity_signal_decay=0.8,
        crowding_aversion=0.8,  # avoids crowded inventory
        information_latency_ms=1.0,
    ),
    AgentType.FUNDAMENTAL_LONG: AgentProfile(
        aum_bn=500.0,
        holding_period_days=252.0,
        leverage_limit=1.5,
        capacity_signal_decay=0.1,   # slow decay (fundamental value is large-cap)
        crowding_aversion=0.2,
        information_latency_ms=86400000.0,  # quarterly earnings cycle
    ),
    AgentType.CRISIS_HEDGE: AgentProfile(
        aum_bn=30.0,
        holding_period_days=21.0,
        leverage_limit=5.0,
        capacity_signal_decay=0.4,
        crowding_aversion=0.7,
        information_latency_ms=1000.0,
    ),
}


class InstitutionalAgent:
    """
    L2/L3 agent with risk-constrained portfolio optimization.

    The agent solves a simplified HJB-derived optimal policy:
      π* = argmax E[r·π] - (γ/2)·π'Σπ - λ·‖π - μ^MFG‖² · 1_{crowding}

    where μ^MFG is the mean-field distribution (current market portfolio weights)
    and λ is the crowding aversion parameter from the profile.

    Parameters
    ----------
    agent_type : AgentType
    risk_aversion : γ in [0.5, 5.0], individual risk aversion
    aum_scale : multiplier on default AUM (for firm-specific sizing)
    rng : random state
    """

    def __init__(
        self,
        agent_type: AgentType,
        risk_aversion: float = 2.0,
        aum_scale: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.agent_type = agent_type
        self.profile = PROFILES[agent_type]
        self.risk_aversion = risk_aversion
        self.aum = self.profile.aum_bn * aum_scale
        self.rng = rng or np.random.default_rng()

    def optimal_weights(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        mean_field_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute optimal portfolio weights via mean-variance + crowding penalty.

        Parameters
        ----------
        expected_returns : (n,) vector of expected returns
        cov_matrix : (n, n) return covariance matrix
        mean_field_weights : (n,) market portfolio (crowding reference)

        Returns
        -------
        π* : (n,) portfolio weights (may exceed 1 if leveraged)
        """
        n = len(expected_returns)
        lam = self.profile.crowding_aversion
        gamma = self.risk_aversion

        Sigma_reg = cov_matrix + 1e-6 * np.eye(n)
        inv_Sigma = np.linalg.solve(Sigma_reg, np.eye(n))

        if mean_field_weights is not None and lam > 0.0:
            # MFG-penalized solution: π* = (γΣ + λI)^{-1}(μ + λ·μ^{MFG})
            A = gamma * Sigma_reg + lam * np.eye(n)
            rhs = expected_returns + lam * mean_field_weights
            weights = np.linalg.solve(A, rhs)
        else:
            # Standard Merton solution: π* = (1/γ) Σ^{-1} μ
            weights = (1.0 / gamma) * inv_Sigma @ expected_returns

        # Apply leverage constraint
        gross_leverage = np.sum(np.abs(weights))
        if gross_leverage > self.profile.leverage_limit:
            weights = weights * (self.profile.leverage_limit / gross_leverage)

        return weights

    def capacity_adjusted_alpha(self, raw_alpha: np.ndarray) -> np.ndarray:
        """
        Scale alpha down by capacity (larger AUM → less alpha per unit).

        Models the empirical observation that alpha decays with AUM due to
        market impact: α_effective = α_raw × exp(-decay × log(AUM/AUM_ref))
        """
        aum_ref = 10.0  # $10B reference size
        decay = self.profile.capacity_signal_decay
        scale = np.exp(-decay * np.log(max(self.aum / aum_ref, 0.1)))
        return raw_alpha * scale

    def crowding_score(
        self,
        portfolio: np.ndarray,
        market_portfolio: np.ndarray,
    ) -> float:
        """
        Measure how crowded this portfolio is relative to the market.

        High crowding score → fragility if mean-field distribution shifts.
        Used as the crisis-warning signal Λ_t feeds into (Day 8).
        """
        if np.linalg.norm(market_portfolio) < 1e-10:
            return 0.0
        cos_sim = (portfolio @ market_portfolio) / (
            np.linalg.norm(portfolio) * np.linalg.norm(market_portfolio) + 1e-12
        )
        return float(np.clip(cos_sim, -1.0, 1.0))

    def __repr__(self) -> str:
        return (
            f"InstitutionalAgent({self.agent_type.name}, "
            f"γ={self.risk_aversion}, AUM=${self.aum:.1f}B)"
        )


def build_agent_population(
    composition: dict[AgentType, int],
    risk_aversion_spread: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> list[InstitutionalAgent]:
    """
    Build a heterogeneous population of institutional agents.

    Parameters
    ----------
    composition : {AgentType: count} how many of each type
    risk_aversion_spread : std dev of gamma around the profile default
    rng : random state

    Returns
    -------
    List of InstitutionalAgent (ordered by type, then by index within type)
    """
    rng = rng or np.random.default_rng()
    agents: list[InstitutionalAgent] = []
    for agent_type, count in composition.items():
        for _ in range(count):
            gamma = max(0.5, 2.0 + rng.normal(0, risk_aversion_spread))
            aum_scale = max(0.1, rng.lognormal(0, 0.3))
            agents.append(InstitutionalAgent(agent_type, gamma, aum_scale, rng))
    return agents
