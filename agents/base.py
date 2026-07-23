"""
Base agent interface. All market participants inherit from this.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from state.portfolio import Portfolio
from state.information import InformationSet


@dataclass
class Order:
    agent_id: str
    ticker: str
    side: str          # "buy" | "sell"
    quantity: float
    order_type: str    # "market" | "limit"
    limit_price: Optional[float] = None
    timestamp: Optional[float] = None


class BaseAgent(ABC):
    def __init__(self, agent_id: str, portfolio: Portfolio, info_set: InformationSet):
        self.agent_id = agent_id
        self.portfolio = portfolio
        self.info_set = info_set

    @abstractmethod
    def act(self, timestamp: float) -> List[Order]:
        """Given current information set, return a list of orders."""
        ...

    def update_info(self, market_state) -> None:
        self.info_set = self.info_set.observe(market_state, timestamp=None)
