"""
Parse SEC 13F filings to extract institutional holdings.
Source: https://www.sec.gov/cgi-bin/browse-edgar (free, no API key needed)

13F is filed quarterly by institutions with >$100M AUM.
This gives us the ground-truth state for each institutional agent.
"""
import requests
import pandas as pd
from typing import Optional


SEC_EDGAR_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": "research-bot contact@example.com"}  # SEC requires User-Agent


def get_13f_filings(cik: str, n: int = 4) -> list[dict]:
    """Return the last n 13F filing metadata for a given CIK."""
    url = f"{SEC_EDGAR_BASE}/submissions/CIK{cik.zfill(10)}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accession = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])

    results = []
    for form, acc, date in zip(forms, accession, dates):
        if form == "13F-HR":
            results.append({"accession": acc, "date": date})
        if len(results) >= n:
            break
    return results


def parse_13f_holdings(cik: str, accession: str) -> Optional[pd.DataFrame]:
    """
    Download and parse the holdings table from a 13F filing.
    Returns DataFrame with columns: [issuer, cusip, shares, market_value, put_call].
    """
    acc_clean = accession.replace("-", "")
    index_url = f"{SEC_EDGAR_BASE}/Archives/edgar/data/{cik}/{acc_clean}/index.json"
    resp = requests.get(index_url, headers=HEADERS)
    if resp.status_code != 200:
        return None

    # Find the primary XML document (infotable)
    files = resp.json().get("directory", {}).get("item", [])
    xml_file = next((f["name"] for f in files if "infotable" in f["name"].lower()), None)
    if not xml_file:
        return None

    xml_url = f"{SEC_EDGAR_BASE}/Archives/edgar/data/{cik}/{acc_clean}/{xml_file}"
    xml_resp = requests.get(xml_url, headers=HEADERS)
    # Parse XML into DataFrame — placeholder for full XML parsing
    # Full implementation: use xml.etree.ElementTree to parse <infoTable> entries
    return pd.DataFrame()  # TODO: implement XML parsing
