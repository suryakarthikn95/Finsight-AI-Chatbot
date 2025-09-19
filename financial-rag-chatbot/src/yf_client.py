# src/yf_client.py
from __future__ import annotations
import math
import datetime as dt
import pandas as pd
import yfinance as yf

# Minimal symbol normalization for Yahoo tickers
def _normalize_for_yf(symbol: str) -> str:
    s = symbol.upper().strip()
    # Common market suffix mappings
    s = s.replace(".NSE", ".NS").replace(".BSE", ".BO")
    s = s.replace(".SAO", ".SA")  # São Paulo
    return s

def get_quote_yf(symbol: str) -> pd.DataFrame:
    """
    Return a 1-row DataFrame with columns:
      symbol, open, high, low, price, volume, latest_trading_day, previous_close, change, change_percent
    """
    sy = _normalize_for_yf(symbol)
    t = yf.Ticker(sy)
    try:
        info = t.fast_info  # fast path
    except Exception:
        info = {}

    price = (getattr(info, "last_price", None) if hasattr(info, "last_price") else info.get("last_price")) \
            or info.get("last_price", None) or info.get("last_trade") or info.get("last")
    if price is None:
        # Try slower fallback via history (last close)
        hist = t.history(period="5d", interval="1d", auto_adjust=False)
        if hist.empty:
            return pd.DataFrame()
        price = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else math.nan
        vol  = int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0
        ltd  = hist.index[-1].date().isoformat()
        chg  = price - prev if prev == prev else math.nan
        chg_pct = (chg / prev * 100) if prev and prev == prev else math.nan
        row = {
            "symbol": sy,
            "open": float(hist["Open"].iloc[-1]),
            "high": float(hist["High"].iloc[-1]),
            "low":  float(hist["Low"].iloc[-1]),
            "price": price,
            "volume": vol,
            "latest_trading_day": ltd,
            "previous_close": prev,
            "change": chg,
            "change_percent": chg_pct,
        }
        return pd.DataFrame([row])

    prev = info.get("previous_close")
    chg = (price - prev) if (prev not in (None, 0)) else math.nan
    chg_pct = (chg / prev * 100) if (prev not in (None, 0)) else math.nan
    ltd = dt.date.today().isoformat()

    row = {
        "symbol": sy,
        "open": float(info.get("open", math.nan)) if info.get("open") is not None else math.nan,
        "high": float(info.get("day_high", math.nan)) if info.get("day_high") is not None else math.nan,
        "low":  float(info.get("day_low", math.nan)) if info.get("day_low") is not None else math.nan,
        "price": float(price),
        "volume": int(info.get("last_volume") or 0),
        "latest_trading_day": ltd,
        "previous_close": float(prev) if prev is not None else math.nan,
        "change": float(chg) if chg == chg else math.nan,
        "change_percent": float(chg_pct) if chg_pct == chg_pct else math.nan,
    }
    return pd.DataFrame([row])

def get_daily_yf(symbol: str, max_points: int = 120) -> pd.DataFrame:
    """
    Daily OHLCV newest-first: columns = date, open, high, low, close, volume
    """
    sy = _normalize_for_yf(symbol)
    t = yf.Ticker(sy)
    hist = t.history(period="6mo", interval="1d", auto_adjust=False)
    if hist.empty:
        return pd.DataFrame()
    df = hist.tail(max_points).copy()
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df["date"] = df.index.date.astype(str)
    df = df[["date","open","high","low","close","volume"]]
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df