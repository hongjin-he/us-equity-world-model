"""
World model simulation engine.
Drives the event loop: tick → agents observe → agents act → orders match → state updates.
"""
from typing import List
from agents.base import BaseAgent, Order
from world.matching import OrderBook
from world.clock import MarketClock


class WorldEngine:
    def __init__(self, agents: List[BaseAgent], tickers: List[str], clock: MarketClock):
        self.agents = agents
        self.tickers = tickers
        self.clock = clock
        self.order_books = {t: OrderBook(t) for t in tickers}
        self.price_history = {t: [] for t in tickers}

    def step(self, timestamp: float) -> dict:
        all_orders: List[Order] = []

        # 1. Each agent observes the current market state
        for agent in self.agents:
            agent.update_info(market_state=self._get_market_state())

        # 2. Each agent decides and submits orders
        for agent in self.agents:
            orders = agent.act(timestamp)
            all_orders.extend(orders)

        # 3. Route orders to the right order book and match
        fills = {}
        for order in all_orders:
            book = self.order_books.get(order.ticker)
            if book:
                fill = book.submit(order)
                if fill:
                    fills.setdefault(order.ticker, []).append(fill)

        # 4. Record clearing prices
        prices = {}
        for ticker, book in self.order_books.items():
            prices[ticker] = book.last_price
            self.price_history[ticker].append((timestamp, book.last_price))

        return prices

    def run(self, n_steps: int) -> dict:
        results = []
        for t in self.clock.timestamps(n_steps):
            prices = self.step(t)
            results.append({"timestamp": t, "prices": prices})
        return results

    def _get_market_state(self):
        return {"prices": {t: self.order_books[t].last_price for t in self.tickers}}
