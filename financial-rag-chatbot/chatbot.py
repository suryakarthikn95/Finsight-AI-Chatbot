# chatbot.py — FinSight AI (Gemini-style header w/o hero input) + Theme Toggle + fast Yahoo/TD router + cached fallbacks
from __future__ import annotations
import os, json, re, time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Data providers & router
from src.td_client import search_symbol_td
from src.router import get_quote_fast, get_daily_fast
from src.cache_store import get_cached_quote, get_cached_daily

# OpenAI (optional: intent/summarize)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ------------------------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="FinSight AI", page_icon="📊", layout="centered")

# ---- Theme Toggle (Dark / Light) ----
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
THEME = st.session_state.theme  # "Dark" or "Light"

PALETTES = {
    "Dark": {
        "bg": "#0B0D11",
        "panel": "#111318",
        "border": "#2a2d36",
        "text": "#E8EAED",
        "muted": "#C8CDD7",
        "primary": "#3C82FF",
        "grad1": "#8ab4ff",
        "grad2": "#5391ff",
        "grad3": "#3c82ff",
        "shadow": "0 10px 28px rgba(0,0,0,.35)",
        "chipHover": "#dbe7ff",
    },
    "Light": {
        "bg": "#FFFFFF",
        "panel": "#F3F5F8",
        "border": "#D9DFEA",
        "text": "#111418",
        "muted": "#2F3A4A",
        "primary": "#3366FF",
        "grad1": "#2B77FF",
        "grad2": "#4E8BFF",
        "grad3": "#699CFF",
        "shadow": "0 10px 28px rgba(0,0,0,.10)",
        "chipHover": "#1c3dff",
    },
}
P = PALETTES[THEME]

def render_theme_css():
    css = f"""
    <style>
    /* Layout + background */
    .block-container {{padding-top: 2.5rem; max-width: 900px;}}
    header[data-testid="stHeader"] {{display: none;}}
    body, .block-container {{ background: {P['bg']}; color: {P['text']}; }}

    /* Hero title */
    .finsight-hero h1 {{
      font-size: clamp(34px, 4.5vw, 54px);
      font-weight: 700;
      text-align: center;
      margin: 1.5rem 0 1.25rem 0;
      letter-spacing: .2px;
      background: linear-gradient(180deg, {P['grad1']}, {P['grad2']} 60%, {P['grad3']});
      -webkit-background-clip: text; background-clip: text; color: transparent;
    }}
    .finsight-hero {{display: grid; place-items: center;}}

    /* Hero shell (no input inside, just visual container for spacing) */
    .finsight-search {{
      width: min(860px, 92vw);
      background: {P['panel']};
      border: 1px solid {P['border']};
      border-radius: 16px;
      box-shadow: {P['shadow']};
      padding: .9rem 1rem;
    }}

    /* Chips */
    .chips {{display:flex; gap:.5rem; flex-wrap:wrap; justify-content:center; margin-top:1rem;}}
    .chip {{
      background:{P['panel']}; border:1px solid {P['border']};
      color:{P['text']}; padding:.45rem .9rem; border-radius:999px;
      font-size:14px; cursor:pointer; transition: all .15s ease;
    }}
    .chip:hover {{border-color:{P['primary']}; color:{P['chipHover']};}}

    /* Chat input placeholder */
    [data-testid="stChatInput"] textarea::placeholder {{opacity:.9;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

render_theme_css()

def finsight_gemini_header() -> None:
    """Render Gemini-style hero (title only, no search bar, no chips)."""
    st.markdown('<div class="finsight-hero"><h1>Hi, I’m FinSight.AI</h1></div>', unsafe_allow_html=True)

# JS: clicking a chip writes into the bottom chat input (user still presses Enter)
st.markdown("""
<script>
const chips = [...document.querySelectorAll('.chip')];
const setChat = (txt) => {
  const ta = window.parent.document.querySelector('[data-testid="stChatInput"] textarea');
  if (ta){ ta.value = txt; ta.dispatchEvent(new Event('input', {bubbles:true})); ta.focus(); }
}
chips.forEach((c)=> c.addEventListener('click', ()=> setChat(c.innerText)));
</script>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Options")
    theme_choice = st.radio("Theme", ["Dark", "Light"], index=0 if THEME=="Dark" else 1, horizontal=True)
    if theme_choice != THEME:
        st.session_state.theme = theme_choice
        st.rerun()

    fast_mode = st.checkbox("⚡ Fast mode (recommended)", value=True, help="Skips LLM summary and trims timeouts.")
    show_llm_summary = st.checkbox("Let OpenAI also summarize results", value=False if fast_mode else True)
    if st.button("Clear caches"):
        try: st.cache_data.clear()
        except Exception: pass
        st.session_state.pop("_NAME_CACHE", None)
        st.success("Caches cleared.")

# ------------------------------------------------------------------------------------
# Intent / parsing helpers
# ------------------------------------------------------------------------------------
@dataclass
class ParsedIntent:
    intent: Literal["quote", "daily"]
    symbol: str
    adjusted: bool = False
    lookback_days: int = 5
    chosen_note: str = ""

INTENT_SYSTEM_PROMPT = """You are a parsing assistant. Extract a structured intent from the user's question about stocks.
Return STRICT JSON with keys: intent ('quote' or 'daily'), symbol (ticker or company name), adjusted (true/false), lookback_days (integer).
- 'latest price/info/quote' => intent='quote'
- 'performance/trend/this week/last N days' => intent='daily'
If the user provides a company NAME (not ticker), return it in 'symbol' as-is; the app will resolve it.
Return ONLY JSON.
"""

COMMON_TICKERS = {"AAPL","MSFT","TSLA","AMZN","GOOGL","META","NVDA","NFLX","INTC","IBM","ORCL"}

def openai_client_or_none() -> Optional["OpenAI"]:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return None
    try: return OpenAI()
    except Exception: return None

def looks_like_ticker(text: str) -> bool:
    t = text.strip().upper()
    return t.isalnum() and (1 <= len(t) <= 6) and t == text.strip() and " " not in text

# ------------------------------------------------------------------------------------
# Caches (Streamlit wrappers around router/search)
# ------------------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def cached_search_symbol(q: str) -> pd.DataFrame:
    return search_symbol_td(q)

@st.cache_data(ttl=45, show_spinner=False)
def cached_quote_fast(sym: str):
    return get_quote_fast(sym)

@st.cache_data(ttl=180, show_spinner=False)
def cached_daily_fast(sym: str):
    return get_daily_fast(sym)

# ------------------------------------------------------------------------------------
# Name → ticker resolution (fast, robust)
# ------------------------------------------------------------------------------------
if "_NAME_CACHE" not in st.session_state:
    st.session_state._NAME_CACHE = {}
_NAME_CACHE: Dict[str, str] = st.session_state._NAME_CACHE

ALIAS_MAP = {
    "infosys": "INFY",          # US ADR (set to "INFY.NS" if you prefer NSE)
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "hdfc": "HDFCBANK.NS",
    "meta": "META", "facebook": "META",
    "apple": "AAPL", "microsoft": "MSFT",
    "alphabet": "GOOGL", "google": "GOOGL",
    "amazon": "AMZN", "tesla": "TSLA", "nvidia": "NVDA",
}

STOPWORDS = {
    "latest","price","quote","today","stock","share","shares","trend","show","get",
    "this","that","for","me","of","the","last","past","days","day","week","weeks",
    "month","months","year","years","performance","info","information","on","give"
}

def _extract_company_hint(text: str) -> str:
    toks = re.findall(r"[A-Za-z][A-Za-z\.-]{1,}", text)
    toks = [t.strip(".-") for t in toks]
    candidates = [t for t in toks if t.lower() not in STOPWORDS]
    candidates.sort(key=lambda s: (-len(s), s))
    return candidates[0].lower() if candidates else text.strip().lower()

def resolve_symbol_to_ticker(raw: str) -> Tuple[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return "", "Please provide a company name or ticker."

    lc_raw = raw.lower()
    if lc_raw in _NAME_CACHE:
        return _NAME_CACHE[lc_raw], ""

    if looks_like_ticker(raw) and raw.upper() in COMMON_TICKERS:
        _NAME_CACHE[lc_raw] = raw.upper()
        return raw.upper(), ""

    hint = _extract_company_hint(raw)
    if hint in ALIAS_MAP:
        _NAME_CACHE[lc_raw] = ALIAS_MAP[hint]
        return ALIAS_MAP[hint], f"Resolved **{raw}** via alias to **{ALIAS_MAP[hint]}**."

    df = cached_search_symbol(hint)
    if df.empty and hint != lc_raw:
        df = cached_search_symbol(raw)

    if df.empty:
        if looks_like_ticker(raw):
            _NAME_CACHE[lc_raw] = raw.upper()
            return raw.upper(), ""
        return "", f"I couldn’t find a ticker for **{raw}** via symbol search."

    best = df.iloc[0]
    symbol = str(best.get("symbol", "")).upper()
    name = best.get("name", "") or best.get("instrument_name", "")
    exch = best.get("exchange", "")
    country = best.get("country", "")
    note = f"Interpreted **{raw}** as **{symbol}** ({name}, {exch or country})."

    alts = []
    for _, r in df.iloc[1:3].iterrows():
        alts.append(f"{str(r.get('symbol','')).upper()} ({r.get('name','') or r.get('instrument_name','')}, {r.get('exchange','') or r.get('country','')})")
    if alts:
        note += " Alternatives: " + "; ".join(alts)

    _NAME_CACHE[lc_raw] = symbol
    return symbol, note

# ------------------------------------------------------------------------------------
# Parsing (local-first; LLM only if complex)
# ------------------------------------------------------------------------------------
@dataclass
class ParsedIntent:
    intent: Literal["quote", "daily"]
    symbol: str
    adjusted: bool = False
    lookback_days: int = 5
    chosen_note: str = ""

AMBIG_PAT = re.compile(r"\b(compare|vs|versus|which|better|analy[sz]e|explain|why)\b", re.I)

def openai_client_or_none() -> Optional["OpenAI"]:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return None
    try: return OpenAI()
    except Exception: return None

def parse_intent_with_llm(query: str, timeout_s: int = 10) -> Optional[ParsedIntent]:
    client = openai_client_or_none()
    if client is None:
        return None
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        if time.time() - t0 > timeout_s:
            return None
        data = json.loads(resp.choices[0].message.content)
        intent = data.get("intent", "quote")
        sym_raw = (data.get("symbol") or "").strip() or query
        adjusted = bool(data.get("adjusted", False))
        lookback_days = int(data.get("lookback_days", 5))
        if intent not in ("quote", "daily"):
            intent = "quote"
        symbol, note = resolve_symbol_to_ticker(sym_raw)
        return ParsedIntent(intent=intent, symbol=symbol, adjusted=adjusted, lookback_days=lookback_days, chosen_note=note)
    except Exception:
        return None

def naive_parse_and_resolve(query: str) -> ParsedIntent:
    intent = "daily"
    if re.search(r"(latest|price|quote|today)", query, flags=re.I):
        intent = "quote"
    m_days = re.search(r"last\s+(\d+)\s+(day|days|d|trading days|sessions)", query, flags=re.I)
    lookback = int(m_days.group(1)) if m_days else (5 if re.search(r"week", query, flags=re.I) else 5)
    symbol, note = resolve_symbol_to_ticker(query)
    return ParsedIntent(intent=intent, symbol=symbol, adjusted=False, lookback_days=max(2, lookback), chosen_note=note)

def parse_intent_smart(query: str) -> ParsedIntent:
    naive = naive_parse_and_resolve(query)
    if naive.symbol and re.search(r"(latest|price|quote|today|last\s+\d+\s+(day|days|d)|week)", query, re.I):
        return naive
    if AMBIG_PAT.search(query):
        intent = parse_intent_with_llm(query, timeout_s=6 if True else 12)
        return intent or naive
    return naive

# ------------------------------------------------------------------------------------
# Chat state + header
# ------------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # start with no bubbles

# Always show the hero (no input)
finsight_gemini_header()

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Single input: chat bar at the bottom
chat_ph = "Ask anything: “AAPL latest price”, “trend for Reliance last 5 days”, “quote META”, “search Infosys”"
user_msg = st.chat_input(chat_ph)

# ------------------------------------------------------------------------------------
# Fetchers (race + cached fallback)
# ------------------------------------------------------------------------------------
def fetch_data(intent: ParsedIntent) -> Tuple[str, Optional[pd.DataFrame]]:
    symbol = intent.symbol
    if not symbol:
        return intent.chosen_note or "I couldn’t resolve a ticker from your query.", None

    # QUOTE
    if intent.intent == "quote":
        try:
            df, provider = cached_quote_fast(symbol)
        except Exception:
            df, provider = None, None

        if df is None or df.empty:
            stale = get_cached_quote(symbol)
            if stale:
                row = stale["row"]
                ts = datetime.utcfromtimestamp(stale["ts"]).strftime("%Y-%m-%d %H:%M:%SZ")
                df = pd.DataFrame([row])
                msg = (
                    (intent.chosen_note + "\n\n") if intent.chosen_note else ""
                ) + (
                    f"**{row.get('symbol', symbol)} — Latest Known Quote** *(cached @ {ts} UTC)*\n\n"
                    f"- Price: **{row.get('price', float('nan')):,.2f}**\n"
                    f"- Change: {row.get('change', float('nan')):,.2f} ({row.get('change_percent', float('nan')):.2f}%)\n"
                    f"- Previous close: {row.get('previous_close', float('nan')):,.2f}\n"
                    f"- Latest trading day: {row.get('latest_trading_day','—')}\n"
                    f"\n_Real-time providers were slow; showing last known cached data._"
                )
                return msg, df
            return f"I couldn't fetch a quote for **{symbol}** and no cached data was available.", None

        row = df.iloc[0]
        msg = (
            (intent.chosen_note + "\n\n") if intent.chosen_note else ""
        ) + (
            f"**{row.get('symbol', symbol)} — Latest Quote** *(via {provider})*\n\n"
            f"- Price: **{row.get('price', float('nan')):,.2f}**\n"
            f"- Change: {row.get('change', float('nan')):,.2f} ({row.get('change_percent', float('nan')):.2f}%)\n"
            f"- Previous close: {row.get('previous_close', float('nan')):,.2f}\n"
            f"- Latest trading day: {row.get('latest_trading_day','—')}\n"
        )
        return msg, df

    # DAILY
    else:
        try:
            df, provider = cached_daily_fast(symbol)
        except Exception:
            df, provider = None, None

        if df is None or df.empty:
            stale = get_cached_daily(symbol)
            if stale:
                rows = stale["rows"]
                ts = datetime.utcfromtimestamp(stale["ts"]).strftime("%Y-%m-%d %H:%M:%SZ")
                df_full = pd.DataFrame(rows)
                df_disp = df_full.head(max(2, intent.lookback_days))
                latest_close = pd.to_numeric(df_disp.iloc[0]["close"], errors="coerce")
                earliest_close = pd.to_numeric(df_disp.iloc[-1]["close"], errors="coerce")
                pct = ((latest_close - earliest_close) / earliest_close) * 100 if pd.notna(latest_close) and pd.notna(earliest_close) else None
                msg = (intent.chosen_note + "\n\n") if intent.chosen_note else ""
                msg += f"**{symbol} — Last {len(df_disp)} trading days** *(cached @ {ts} UTC)*\n\n"
                if pct is not None:
                    msg += f"- Change over period: **{pct:.2f}%**\n"
                msg += "- Showing cached rows below.\n\n_Real-time providers were slow; showing last known cached data._"
                return msg, df_disp
            return f"I couldn't fetch daily data for **{symbol}** and no cached data was available.", None

        df_disp = df.head(max(2, intent.lookback_days))
        latest_close = pd.to_numeric(df_disp.iloc[0]["close"], errors="coerce")
        earliest_close = pd.to_numeric(df_disp.iloc[-1]["close"], errors="coerce")
        pct = ((latest_close - earliest_close) / earliest_close) * 100 if pd.notna(latest_close) and pd.notna(earliest_close) else None
        msg = (intent.chosen_note + "\n\n") if intent.chosen_note else ""
        msg += f"**{symbol} — Last {len(df_disp)} trading days** *(via {provider})*\n\n"
        if pct is not None:
            msg += f"- Change over period: **{pct:.2f}%**\n"
        msg += "- Showing top rows below."
        return msg, df_disp

# ------------------------------------------------------------------------------------
# Chat loop
# ------------------------------------------------------------------------------------
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("🔎 Understanding your request…"):
            intent = parse_intent_smart(user_msg)
            if not intent.symbol:
                st.markdown(intent.chosen_note or "I couldn't resolve a ticker from your query.")
                st.stop()

        with st.spinner(f"📡 Fetching data for {intent.symbol}…"):
            data_text, df = fetch_data(intent)

        final_text = data_text
        if (not fast_mode) and show_llm_summary and df is not None:
            try:
                client = openai_client_or_none()
                if client:
                    if intent.intent == "quote":
                        row = df.iloc[0].to_dict()
                        prompt = "Summarize this stock quote in ≤2 sentences:\n\n" + json.dumps(row, indent=2)
                    else:
                        sample = df.head(10).to_dict(orient="records")
                        prompt = (f"Summarize the trend for {intent.symbol} over the last {len(df)} rows. "
                                  "Include a simple percent change if applicable. Keep it under 2 sentences.\n\n"
                                  f"Data:\n{json.dumps(sample, indent=2)}")
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": "Be concise and factual."},
                                  {"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    summary = resp.choices[0].message.content
                    if summary:
                        final_text = f"{final_text}\n\n**Summary:** {summary}"
            except Exception:
                st.caption("Summary skipped.")

        st.markdown(final_text)
        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True)

    st.session_state.messages.append({"role": "assistant", "content": final_text})