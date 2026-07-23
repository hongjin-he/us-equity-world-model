"""§2.1.4 — News headlines + sentence-transformer embeddings (local, free)."""
import os
import numpy as np
import requests


NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "[YOUR_KEY_HERE]")


def get_headlines(query: str, from_date: str) -> list[str]:
    url = "https://newsapi.org/v2/everything"
    resp = requests.get(url, params={
        "q": query, "from": from_date,
        "apiKey": NEWSAPI_KEY, "language": "en", "sortBy": "publishedAt",
        "pageSize": 100,
    })
    resp.raise_for_status()
    return [a["title"] for a in resp.json().get("articles", [])]


def embed_headlines(headlines: list[str]) -> np.ndarray:
    """
    Embed headlines using sentence-transformers (runs locally, no API cost).
    Returns [384] vector (mean-pooled over headlines).
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if not headlines:
        return np.zeros(384)
    vecs = model.encode(headlines, normalize_embeddings=True)
    return vecs.mean(axis=0)
