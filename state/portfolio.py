from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Portfolio:
    agent_id: str
    cash: float = 0.0
    holdings: dict = field(default_factory=dict)   # {ticker: shares}
    leverage: float = 1.0
    gross_notional: float = 0.0

    def market_value(self, prices: dict) -> float:
        return sum(self.holdings.get(t, 0) * prices.get(t, 0) for t in self.holdings) + self.cash

    def weight(self, ticker: str, prices: dict) -> float:
        mv = self.market_value(prices)
        if mv == 0:
            return 0.0
        return self.holdings.get(ticker, 0) * prices.get(ticker, 0) / mv
