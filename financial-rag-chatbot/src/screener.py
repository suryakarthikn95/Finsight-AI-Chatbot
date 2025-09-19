# src/screener.py
import pandas as pd
import yfinance as yf
from datetime import datetime

# Example universes
UNIVERSES = {
    "dax": {
        "label": "DAX 40 (Xetra)",
        "tickers": ["AIR.DE","VOW3.DE","DBK.DE","PAH3.DE","ENR.DE","BMW.DE","DTG.DE","BAS.DE","HEI.DE","CON.DE"]
    },
    "ftse": {
        "label": "FTSE 100",
        "tickers": ["AZN.L","HSBA.L","BP.L","GSK.L","ULVR.L"]
    },
    "nifty": {
        "label": "NIFTY 50",
        "tickers": ["INFY.NS","RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS"]
    }
}

def resolve_universe(query: str):
    q = query.lower()
    for key, meta in UNIVERSES.items():
        if key in q or meta["label"].split()[0].lower() in q:
            return key, meta["tickers"], meta["label"]
    return "", [], ""

def screen_universe(tickers, mode="gainers", top_n=12):
    rows = []
    for t in tickers:
        try:
            data = yf.Ticker(t).history(period="5d")
            if data.empty:
                continue
            last = data["Close"].iloc[-1]
            prev = data["Close"].iloc[-2] if len(data) > 1 else last
            change_today = (last - prev) / prev * 100 if prev else 0
            change_5d = (last - data["Close"].iloc[0]) / data["Close"].iloc[0] * 100
            vol_today = data["Volume"].iloc[-1]
            rows.append({
                "symbol": t, "last": round(last,2),
                "pct_change_today": round(change_today,3),
                "pct_change_5d": round(change_5d,3),
                "volume_today": int(vol_today),
                "as_of": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df, "No data"

    if mode == "losers":
        df = df.sort_values("pct_change_today").head(top_n)
        title = "Top losers"
    elif mode == "active":
        df = df.sort_values("volume_today", ascending=False).head(top_n)
        title = "Most active"
    else:
        df = df.sort_values("pct_change_today", ascending=False).head(top_n)
        title = "Top gainers"
    return df.reset_index(drop=True), title