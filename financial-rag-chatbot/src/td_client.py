
from __future__ import annotations
import os, math
from typing import Optional, Dict, Any
import httpx
import pandas as pd
from dotenv import load_dotenv

_TD_BASE = "https://api.twelvedata.com"

class TwelveDataError(Exception):
    pass

_client: Optional[httpx.Client] = None

def _get_client(timeout: float = 2.5) -> httpx.Client:
    """Persistent HTTP/1.1 client with tight timeouts for snappier failure."""
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=_TD_BASE,
            timeout=timeout,
            headers={"User-Agent": "financial-rag/1.0"},
        )
    else:
        _client.timeout = timeout
    return _client

def _get_api_key() -> str:
    load_dotenv()
    key = os.getenv("TWELVEDATA_API_KEY")
    if not key:
        raise TwelveDataError("Missing TWELVEDATA_API_KEY in environment (.env).")
    return key

def _request(path: str, params: Dict[str, Any], timeout: float = 2.5) -> Dict[str, Any]:
    client = _get_client(timeout)
    r = client.get(path, params=params)
    try:
        data = r.json()
    except Exception:
        raise TwelveDataError(f"Non-JSON response, status {r.status_code}")
    if isinstance(data, dict) and data.get("code"):
        raise TwelveDataError(f"{data.get('code')}: {data.get('message')}")
    return data

# ========================
# API wrappers
# ========================

def search_symbol_td(keywords: str, timeout: float = 3.0) -> pd.DataFrame:
    api_key = _get_api_key()
    data = _request("symbol_search", {"symbol": keywords, "apikey": api_key}, timeout=timeout)
    if "data" not in data:
        return pd.DataFrame()
    return pd.DataFrame(data["data"])

def get_quote_td(symbol: str, timeout: float = 4.0) -> pd.DataFrame:
    """
    Latest quote. Normalizes to:
      symbol, open, high, low, price, volume, latest_trading_day, previous_close, change, change_percent
    """
    api_key = _get_api_key()
    data = _request("quote", {"symbol": symbol, "apikey": api_key}, timeout=timeout)

    # Accept either 'price' or 'close'
    raw_price = data.get("price", None)
    if raw_price in (None, "", "NaN"):
        raw_price = data.get("close", None)

    if raw_price in (None, "", "NaN"):
        raise TwelveDataError(f"Unexpected quote response keys: {list(data.keys())}")

    def fnum(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    row = {
        "symbol": (data.get("symbol") or symbol).upper(),
        "open": fnum(data.get("open")),
        "high": fnum(data.get("high")),
        "low": fnum(data.get("low")),
        "price": fnum(raw_price),
        "volume": int(float(data.get("volume") or 0)),
        "latest_trading_day": data.get("datetime") or "",
        "previous_close": fnum(data.get("previous_close")),
        "change": fnum(data.get("change")),
        "change_percent": fnum(data.get("percent_change")),
    }

    return pd.DataFrame([row])

def get_daily_td(symbol: str, output_size: str = "compact", timeout: float = 5.0) -> pd.DataFrame:
    """
    Daily OHLCV newest-first. Columns: date, open, high, low, close, volume
    """
    api_key = _get_api_key()
    out = 100 if output_size == "compact" else 5000
    data = _request(
        "time_series",
        {"symbol": symbol, "interval": "1day", "apikey": api_key, "outputsize": out},
        timeout=timeout,
    )
    if "values" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "date"})
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date", ascending=False).reset_index(drop=True)
