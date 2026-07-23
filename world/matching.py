"""
Price-time priority order book for a single ticker.
"""
import heapq
from dataclasses import dataclass, field
from typing import Optional
from agents.base import Order


@dataclass
class Fill:
    ticker: str
    price: float
    quantity: float
    buyer_id: str
    seller_id: str
    timestamp: float


class OrderBook:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.bids: list = []   # max-heap (negate price)
        self.asks: list = []   # min-heap
        self.last_price: Optional[float] = None
        self._pending_bids = []
        self._pending_asks = []

    def submit(self, order: Order) -> Optional[Fill]:
        if order.order_type == "market":
            return self._match_market(order)
        elif order.order_type == "limit":
            self._add_limit(order)
            return self._try_match()
        return None

    def _match_market(self, order: Order) -> Optional[Fill]:
        # Simplified: match against best opposite side
        if order.side == "buy" and self._pending_asks:
            ask_price, ask_qty, ask_agent = self._pending_asks[0]
            fill_qty = min(order.quantity, ask_qty)
            self.last_price = ask_price
            self._pending_asks[0] = (ask_price, ask_qty - fill_qty, ask_agent)
            if self._pending_asks[0][1] <= 0:
                self._pending_asks.pop(0)
            return Fill(self.ticker, ask_price, fill_qty, order.agent_id, ask_agent, order.timestamp or 0)
        elif order.side == "sell" and self._pending_bids:
            bid_price, bid_qty, bid_agent = self._pending_bids[0]
            fill_qty = min(order.quantity, bid_qty)
            self.last_price = bid_price
            self._pending_bids[0] = (bid_price, bid_qty - fill_qty, bid_agent)
            if self._pending_bids[0][1] <= 0:
                self._pending_bids.pop(0)
            return Fill(self.ticker, bid_price, fill_qty, bid_agent, order.agent_id, order.timestamp or 0)
        return None

    def _add_limit(self, order: Order) -> None:
        if order.side == "buy":
            self._pending_bids.append((order.limit_price, order.quantity, order.agent_id))
            self._pending_bids.sort(key=lambda x: -x[0])
        else:
            self._pending_asks.append((order.limit_price, order.quantity, order.agent_id))
            self._pending_asks.sort(key=lambda x: x[0])

    def _try_match(self) -> Optional[Fill]:
        if self._pending_bids and self._pending_asks:
            bid_price, bid_qty, bid_agent = self._pending_bids[0]
            ask_price, ask_qty, ask_agent = self._pending_asks[0]
            if bid_price >= ask_price:
                fill_price = (bid_price + ask_price) / 2
                fill_qty = min(bid_qty, ask_qty)
                self.last_price = fill_price
                self._pending_bids[0] = (bid_price, bid_qty - fill_qty, bid_agent)
                self._pending_asks[0] = (ask_price, ask_qty - fill_qty, ask_agent)
                self._pending_bids = [(p, q, a) for p, q, a in self._pending_bids if q > 0]
                self._pending_asks = [(p, q, a) for p, q, a in self._pending_asks if q > 0]
                return Fill(self.ticker, fill_price, fill_qty, bid_agent, ask_agent, 0)
        return None
