# src/router.py
from __future__ import annotations
import os, concurrent.futures as fut
from typing import Optional, Tuple
import pandas as pd

from src.td_client import get_quote_td, get_daily_td
from src.yf_client import get_quote_yf, get_daily_yf
from src.av_client import get_global_quote as av_quote, get_daily as av_daily  # optional
from src.cache_store import cache_quote, cache_daily

def _valid_quote(df: Optional[pd.DataFrame]) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty and "price" in df.columns

def _valid_daily(df: Optional[pd.DataFrame]) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty and {"date", "close"}.issubset(df.columns)

def get_quote_fast(symbol: str, timeout_each: float = 5.0) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    # Yahoo tends to respond fastest → try it first in the race
    def yf():
        try: return get_quote_yf(symbol)
        except Exception: return None
    def td():
        try: return get_quote_td(symbol)
        except Exception: return None
    def av():
        try: return av_quote(symbol)
        except Exception: return None

    providers = [("yahoo", yf), ("twelve_data", td)]
    if os.getenv("ENABLE_ALPHA_VANTAGE"):
        providers.append(("alpha_vantage", av))

    with fut.ThreadPoolExecutor(max_workers=len(providers)) as ex:
        futures = {ex.submit(fn): name for name, fn in providers}
        for done in fut.as_completed(futures):
            name = futures[done]
            df = done.result()
            if _valid_quote(df):
                try: cache_quote(symbol, df.iloc[0].to_dict())
                except Exception: pass
                return df, name
    return None, None

def get_daily_fast(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    def yf():
        try: return get_daily_yf(symbol, max_points=120)
        except Exception: return None
    def td():
        try: return get_daily_td(symbol, output_size="compact")
        except Exception: return None
    def av():
        try: return av_daily(symbol, output_size="compact", adjusted=False)
        except Exception: return None

    providers = [("yahoo", yf), ("twelve_data", td)]
    if os.getenv("ENABLE_ALPHA_VANTAGE"):
        providers.append(("alpha_vantage", av))

    with fut.ThreadPoolExecutor(max_workers=len(providers)) as ex:
        futures = {ex.submit(fn): name for name, fn in providers}
        for done in fut.as_completed(futures):
            name = futures[done]
            df = done.result()
            if _valid_daily(df):
                try: cache_daily(symbol, df.to_dict(orient="records"))
                except Exception: pass
                return df, name
    return None, None