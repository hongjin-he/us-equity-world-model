"""§5.2 — Execution interface: Alpaca Paper Trading API (no real capital)."""
import os
import numpy as np

ALPACA_KEY    = os.getenv("ALPACA_KEY",    "[YOUR_KEY_HERE]")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "[YOUR_SECRET_HERE]")
ALPACA_BASE   = "https://paper-api.alpaca.markets"   # PAPER endpoint


def _get_api():
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE, api_version="v2")


def submit_orders(delta_shares: dict[str, float], min_lot: int = 1) -> list:
    api = _get_api()
    orders = []
    for ticker, qty in delta_shares.items():
        if abs(qty) < min_lot:
            continue
        # Pre-screen: skip if stock halted (order would be rejected)
        try:
            asset = api.get_asset(ticker)
            if not asset.tradable:
                continue
        except Exception:
            continue
        order = api.submit_order(
            symbol=ticker,
            qty=abs(int(qty)),
            side="buy" if qty > 0 else "sell",
            type="market",
            time_in_force="day",
        )
        orders.append(order)
    return orders


def get_positions() -> dict[str, float]:
    return {p.symbol: float(p.qty) for p in _get_api().list_positions()}


def get_account() -> dict:
    acc = _get_api().get_account()
    return {"equity": float(acc.equity), "cash": float(acc.cash)}
