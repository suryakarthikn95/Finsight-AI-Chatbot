# src/av_client.py
from __future__ import annotations
import os
import time
from typing import Literal, Optional, Dict, Any
import requests
import pandas as pd
from dotenv import load_dotenv

_AV_BASE = "https://www.alphavantage.co/query"

class AlphaVantageError(Exception):
    pass

def _get_api_key() -> str:
    load_dotenv()
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise AlphaVantageError("Missing ALPHAVANTAGE_API_KEY in environment (.env).")
    return key

def _request(params: Dict[str, Any], attempts: int = 3, delays = (10, 20, 30)) -> Dict[str, Any]:
    """
    Basic robust requester that handles free-tier throttling ("Note"/"Information") with retries.
    """
    for i in range(attempts):
        resp = requests.get(_AV_BASE, params=params, timeout=30)
        data = resp.json()
        if "Error Message" in data:
            raise AlphaVantageError(data["Error Message"])
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information")
            if i < attempts - 1:
                time.sleep(delays[i])
                continue
            raise AlphaVantageError(f"Throttled/Premium message: {msg}")
        return data
    raise AlphaVantageError("Gave up after retries.")

def get_global_quote(symbol: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Returns a 1-row DataFrame with latest quote for `symbol`.
    Columns (normalized): symbol, open, high, low, price, volume, previous close, change, change percent, latest trading day.
    """
    api_key = api_key or _get_api_key()
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key}
    data = _request(params)

    key = "Global Quote"
    if key not in data or not data[key]:
        raise AlphaVantageError(f"Unexpected response: keys={list(data.keys())}")

    raw = data[key]
    clean = { (k.split(". ")[1] if ". " in k else k): v for k, v in raw.items() }
    df = pd.DataFrame([clean])

    # normalize types
    for col in ["price","open","high","low","previous close","change","volume"]:
        if col in df.columns:
            if col == "volume":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    if "change percent" in df.columns:
        df["change percent"] = (
            df["change percent"].astype(str).str.replace("%","", regex=False)
        )
        df["change percent"] = pd.to_numeric(df["change percent"], errors="coerce")
    # rename for friendliness
    df = df.rename(columns={"latest trading day":"latest_trading_day", "previous close":"previous_close", "change percent":"change_percent"})
    return df

def get_daily(
    symbol: str,
    output_size: Literal["compact","full"]="compact",
    adjusted: bool = False,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Returns a DataFrame of daily bars (newest first).
    If `adjusted=True`, uses TIME_SERIES_DAILY_ADJUSTED.
    """
    api_key = api_key or _get_api_key()
    fn = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
    params = {"function": fn, "symbol": symbol, "apikey": api_key, "outputsize": output_size, "datatype":"json"}
    data = _request(params, attempts=3, delays=(20,40,60))

    key = "Time Series (Daily)"
    if key not in data:
        raise AlphaVantageError(f"Unexpected response: keys={list(data.keys())}")

    ts = data[key]
    df = (
        pd.DataFrame.from_dict(ts, orient="index")
        .rename(columns=lambda c: c.split(". ")[1] if ". " in c else c)
        .reset_index()
        .rename(columns={"index": "date"})
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )
    # cast numeric columns if present
    numeric_cols = ["open","high","low","close","adjusted close","volume","dividend amount","split coefficient"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df