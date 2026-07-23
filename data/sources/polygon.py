"""§2.1.1 — Market microstructure data via Polygon.io (OHLCV + trades + quotes)."""
import os, asyncio, json
import requests
import websockets
from typing import AsyncGenerator


POLYGON_KEY = os.getenv("POLYGON_KEY", "[YOUR_KEY_HERE]")
BASE = "https://api.polygon.io/v2"


def get_aggs(ticker: str, from_date: str, to_date: str,
             multiplier: int = 1, timespan: str = "minute") -> dict:
    """Fetch OHLCV bars. Free tier: 5 req/min, 2yr history."""
    url = f"{BASE}/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    resp = requests.get(url, params={"apiKey": POLYGON_KEY, "adjusted": "true"})
    resp.raise_for_status()
    return resp.json()


def get_ticker_details(ticker: str) -> dict:
    """Shares outstanding, market cap — feeds κ_t in state space."""
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
    resp = requests.get(url, params={"apiKey": POLYGON_KEY})
    resp.raise_for_status()
    return resp.json().get("results", {})


async def stream_trades(tickers: list[str]) -> AsyncGenerator[dict, None]:
    """Real-time trade stream via WebSocket."""
    async with websockets.connect("wss://socket.polygon.io/stocks") as ws:
        await ws.send(json.dumps({"action": "auth", "params": POLYGON_KEY}))
        sub = ",".join(f"T.{t}" for t in tickers)
        await ws.send(json.dumps({"action": "subscribe", "params": sub}))
        async for msg in ws:
            for item in json.loads(msg):
                yield item
