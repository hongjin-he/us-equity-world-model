"""§2.1.3 — Macro state variables via FRED (Federal Reserve Economic Data). Free, no limits."""
import os
import requests
import pandas as pd


FRED_KEY = os.getenv("FRED_KEY", "[YOUR_KEY_HERE]")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Core macro series feeding the market state x_macro
MACRO_SERIES = {
    "DFF":      "Fed Funds Rate",
    "CPIAUCSL": "CPI (YoY inflation proxy)",
    "UNRATE":   "Unemployment Rate",
    "T10Y2Y":   "Yield Curve (10Y-2Y spread)",
    "VIXCLS":   "VIX (behavioral noise proxy λ_η)",
    "M2SL":     "M2 Money Supply",
}


def get_series(series_id: str, start: str = "2010-01-01") -> pd.Series:
    resp = requests.get(FRED_BASE, params={
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "observation_start": start,
    })
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    s = pd.Series(
        {o["date"]: float(o["value"]) for o in obs if o["value"] != "."},
        name=series_id,
    )
    s.index = pd.to_datetime(s.index)
    return s


def get_macro_state(date: str) -> dict:
    """Return latest macro state vector for a given date."""
    state = {}
    for sid in MACRO_SERIES:
        try:
            s = get_series(sid, start=date)
            state[sid] = float(s.iloc[-1]) if len(s) > 0 else 0.0
        except Exception:
            state[sid] = 0.0
    return state
