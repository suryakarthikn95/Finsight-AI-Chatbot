# scripts/test_alpha_vantage.py
import os
import sys
import time
import json
import argparse
import requests
import pandas as pd
from dotenv import load_dotenv

def wait_and_retry_message(msg, attempt, total_attempts, delays):
    if attempt < total_attempts - 1:
        wait_s = delays[attempt]
        print(f"⏳ Throttled: {msg}\n   Waiting {wait_s}s then retrying... ({attempt+1}/{total_attempts})")
        time.sleep(wait_s)
        return True
    else:
        print("❌ Still throttled after retries. Message from API:")
        print(msg)
        return False

def fetch_global_quote(symbol: str, api_key: str, attempts=3):
    """
    Fetches latest quote (price, change, volume, etc.) from Alpha Vantage GLOBAL_QUOTE.
    Returns a 1-row pandas DataFrame.
    """
    url = "https://www.alphavantage.co/query"
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key}
    delays = [10, 20, 30]

    for attempt in range(attempts):
        resp = requests.get(url, params=params, timeout=30)
        try:
            data = resp.json()
        except Exception:
            print("❌ Response was not JSON. Status:", resp.status_code)
            sys.exit(1)

        if "Error Message" in data:
            print("❌ API error:", data["Error Message"])
            sys.exit(1)

        # Throttle / info messages
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information")
            if wait_and_retry_message(msg, attempt, attempts, delays):
                continue
            else:
                sys.exit(1)

        key = "Global Quote"
        if key not in data or not data[key]:
            print("❌ Unexpected response shape. Got keys:", list(data.keys()))
            # Show raw for debugging
            print("Raw response:", json.dumps(data, indent=2)[:800])
            sys.exit(1)

        quote = data[key]
        # Convert to 1-row DataFrame
        clean = {k.split(". ")[1] if ". " in k else k: v for k, v in quote.items()}
        df = pd.DataFrame([clean])
        # Cast numerics
        for col in ["price", "open", "high", "low", "volume", "previous close", "change", "change percent"]:
            if col in df.columns:
                if col == "change percent":
                    # strip % sign
                    df[col] = df[col].str.replace("%", "", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    print("❌ Gave up after retries.")
    sys.exit(1)

def fetch_daily(symbol: str, api_key: str, output_size="compact", attempts=3):
    """
    Simpler daily series (non-adjusted) as a secondary test if you want it.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": output_size,
        "datatype": "json",
    }
    delays = [20, 40, 60]

    for attempt in range(attempts):
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        if "Error Message" in data:
            print("❌ API error:", data["Error Message"])
            sys.exit(1)
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information")
            if wait_and_retry_message(msg, attempt, attempts, delays):
                continue
            else:
                sys.exit(1)
        key = "Time Series (Daily)"
        if key not in data:
            print("❌ Unexpected response shape. Got keys:", list(data.keys()))
            print("Raw response:", json.dumps(data, indent=2)[:800])
            sys.exit(1)
        ts = data[key]
        df = (
            pd.DataFrame.from_dict(ts, orient="index")
            .rename(columns=lambda c: c.split(". ")[1] if ". " in c else c)
            .reset_index()
            .rename(columns={"index": "date"})
            .sort_values("date", ascending=False)
            .reset_index(drop=True)
        )
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    print("❌ Gave up after retries.")
    sys.exit(1)

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Alpha Vantage smoke test")
    parser.add_argument("symbol", nargs="?", default="MSFT")
    parser.add_argument("--mode", choices=["quote", "daily"], default="quote")
    args = parser.parse_args()

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("❌ Missing ALPHAVANTAGE_API_KEY in your .env file.")
        sys.exit(1)

    print(f"▶ Testing Alpha Vantage ({args.mode}) for {args.symbol}...")

    try:
        if args.mode == "quote":
            df = fetch_global_quote(args.symbol, api_key)
            print("✅ Success! Latest quote:\n")
            print(df.to_string(index=False))
        else:
            df = fetch_daily(args.symbol, api_key, output_size="compact")
            print("✅ Success! Top 5 rows:\n")
            print(df.head(5)[["date", "open", "high", "low", "close", "volume"]].to_string(index=False))
    except SystemExit:
        # If throttled or premium-gated, try the public demo key quickly for MSFT to verify wiring
        if os.getenv("ALPHAVANTAGE_API_KEY") != "demo":
            print("\n🔁 Quick connectivity check with demo key for MSFT...")
            try:
                if args.mode == "quote":
                    df = fetch_global_quote("MSFT", "demo")
                    print("✅ Demo key works. Your setup is correct; your personal key is just throttled right now.")
                    print(df.to_string(index=False))
                else:
                    df = fetch_daily("MSFT", "demo", output_size="compact")
                    print("✅ Demo key works. Your setup is correct; your personal key is just throttled right now.")
                    print(df.head(3).to_string(index=False))
            except SystemExit:
                print("❌ Even the demo key failed. Please share the exact output above and we’ll fix it.")
        raise