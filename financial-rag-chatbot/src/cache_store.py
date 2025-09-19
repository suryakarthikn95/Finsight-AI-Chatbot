# src/cache_store.py
from __future__ import annotations
import json, os, time
from typing import Optional, Dict, Any, List

# simple JSON-on-disk cache
_CACHE_PATH = os.path.join(".cache", "market_cache.json")
os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)

def _load() -> Dict[str, Any]:
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save(obj: Dict[str, Any]) -> None:
    tmp = _CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, _CACHE_PATH)

def cache_quote(symbol: str, row: Dict[str, Any]) -> None:
    data = _load()
    now = int(time.time())
    data.setdefault("quotes", {})[symbol.upper()] = {"row": row, "ts": now}
    _save(data)

def get_cached_quote(symbol: str) -> Optional[Dict[str, Any]]:
    data = _load()
    return data.get("quotes", {}).get(symbol.upper())

def cache_daily(symbol: str, rows: List[Dict[str, Any]]) -> None:
    data = _load()
    now = int(time.time())
    data.setdefault("daily", {})[symbol.upper()] = {"rows": rows, "ts": now}
    _save(data)

def get_cached_daily(symbol: str) -> Optional[Dict[str, Any]]:
    data = _load()
    return data.get("daily", {}).get(symbol.upper())