"""§2.2 — Kafka ingestion pipeline. Run as a background service."""
import asyncio, json, os, threading
from kafka import KafkaProducer

from data.sources.polygon import stream_trades
from data.sources.fred import get_macro_state
from data.sources.news import get_headlines, embed_headlines

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BOOTSTRAP],
    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    compression_type="gzip",
)


def produce_trades(tickers: list[str]) -> None:
    async def _run():
        async for msg in stream_trades(tickers):
            producer.send("market_trades", value=msg)
    asyncio.run(_run())


def produce_macro(interval_seconds: int = 3600) -> None:
    import time
    while True:
        state = get_macro_state(date="2010-01-01")
        producer.send("macro_state", value=state)
        time.sleep(interval_seconds)


def produce_news(tickers: list[str], interval_seconds: int = 900) -> None:
    import time
    from datetime import date, timedelta
    while True:
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        for ticker in tickers:
            headlines = get_headlines(ticker, from_date=yesterday)
            emb = embed_headlines(headlines).tolist()
            producer.send("news_embeddings", value={"ticker": ticker, "embedding": emb})
        time.sleep(interval_seconds)


def start_all(tickers: list[str]) -> list[threading.Thread]:
    threads = [
        threading.Thread(target=produce_trades, args=[tickers], daemon=True),
        threading.Thread(target=produce_macro, daemon=True),
        threading.Thread(target=produce_news, args=[tickers], daemon=True),
    ]
    for t in threads:
        t.start()
    return threads
