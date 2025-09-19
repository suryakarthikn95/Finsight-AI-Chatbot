from __future__ import annotations
from typing import List, Tuple, Dict
import re

# --- Curated universes using Yahoo tickers (extend as needed) ---

DAX40: List[str] = [
    "ADS.DE","AIR.DE","ALV.DE","BAS.DE","BAYN.DE","BMW.DE","CON.DE","1COV.DE","DTE.DE",
    "DPW.DE","DTG.DE","DB1.DE","DBK.DE","DWNI.DE","ENR.DE","FME.DE","FRE.DE","HEI.DE",
    "HEN3.DE","HNR1.DE","IFX.DE","LIN.DE","MRK.DE","MTX.DE","MUV2.DE","PAH3.DE","PUM.DE",
    "QIA.DE","RWE.DE","SAP.DE","SIE.DE","SHL.DE","SY1.DE","VOW3.DE","VNA.DE","VNA.DE",
    "ZAL.DE","HFG.DE","BEI.DE","VNA.DE"
]
# Remove dupes:
DAX40 = sorted(list(dict.fromkeys(DAX40)))

FTSE100: List[str] = [
    "AZN.L","SHEL.L","HSBA.L","BP.L","GSK.L","DGE.L","ULVR.L","RIO.L","BATS.L","BHP.L",
    "LLOY.L","BARC.L","VOD.L","BA.L","IMB.L","GLEN.L","NG.L","AV.L","NXT.L","RKT.L",
    "REL.L","NWG.L","TSCO.L","AAL.L","BT-A.L","LSEG.L","STAN.L","SKG.L","WPP.L","RR.L",
    "SBRY.L","SSE.L","SMT.L","MNDI.L","ANTO.L","IHG.L","CRH.L","HLMA.L","HL.L","ABF.L",
    "AUTO.L","CPG.L","PSN.L","PHNX.L","WOSG.L","JD.L","HIK.L","PSN.L","TW.L","SPX.L"
]
FTSE100 = sorted(list(dict.fromkeys(FTSE100)))

CAC40: List[str] = [
    "OR.PA","MC.PA","AIR.PA","BNP.PA","DG.PA","KER.PA","BN.PA","SU.PA","SAF.PA","SAN.PA",
    "ENGI.PA","EL.PA","VIV.PA","ACA.PA","AI.PA","GLE.PA","CAP.PA","HO.PA","STLA.PA","RI.PA",
    "TTE.PA","URW.AS","ORV.PA","STM.PA","RNO.PA","FTI.PA","PUB.PA","ATO.PA","WLN.PA","VIE.PA",
    "SGO.PA","LR.PA","EDF.PA","SW.PA","EN.PA","FR.PA","AIR.PA","ACA.PA","CS.PA","ALO.PA"
]
CAC40 = sorted(list(dict.fromkeys(CAC40)))

NIFTY50: List[str] = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","ITC.NS","BHARTIARTL.NS",
    "SBIN.NS","LT.NS","AXISBANK.NS","KOTAKBANK.NS","LICI.NS","BAJFINANCE.NS","HINDUNILVR.NS",
    "LTIM.NS","MARUTI.NS","ASIANPAINT.NS","WIPRO.NS","HCLTECH.NS","SUNPHARMA.NS","ONGC.NS",
    "NTPC.NS","TITAN.NS","M&M.NS","POWERGRID.NS","ULTRACEMCO.NS","BAJAJFINSV.NS","HEROMOTOCO.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","ADANIENT.NS","ADANIPORTS.NS","BRITANNIA.NS","COALINDIA.NS",
    "BPCL.NS","IOC.NS","TECHM.NS","GRASIM.NS","CIPLA.NS","DRREDDY.NS","SBILIFE.NS","HDFCLIFE.NS",
    "BAJAJ-AUTO.NS","TATAMOTORS.NS","EICHERMOT.NS","DIVISLAB.NS","UPL.NS","HINDALCO.NS","SHREECEM.NS"
]
NIFTY50 = sorted(list(dict.fromkeys(NIFTY50)))

SP500_SAMPLE: List[str] = [
    # Sample to keep fetch fast; replace with full list later if desired.
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","BRK-B","UNH","XOM","JNJ","JPM","V","PG","AVGO","HD","MA","CVX","PFE","KO","PEP","ABBV","COST","ADBE","CSCO","ACN","TMO","DHR","MCD","WMT"
]

NASDAQ100: List[str] = [
    "AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","TSLA","AVGO","PEP","COST","ADBE","NFLX","CSCO",
    "AMD","INTC","QCOM","TXN","AMAT","PYPL","SBUX","HON","GILD","BKNG","ADI","ADP","INTU","AMGN","MDLZ","MU"
]

NIKKEI225: List[str] = [
    "7203.T","9984.T","6758.T","6861.T","6501.T","9020.T","4063.T","6902.T","8035.T","7735.T",
    "7267.T","9432.T","2914.T","6367.T","6981.T","7269.T","8591.T","8001.T","8411.T","4502.T"
]

# --- Aliases: flexible user phrases
ALIASES: Dict[str, List[str]] = {
    "dax": ["dax", "dax40", "xetra", "deutsche börse", "germany index"],
    "ftse100": ["ftse", "ftse100", "ftse 100", "london", "lse 100"],
    "cac40": ["cac", "cac40", "cac 40", "paris", "euronext paris"],
    "nifty50": ["nifty", "nifty 50", "nse", "india nifty", "nifty50"],
    "sp500": ["s&p", "sp500", "s&p 500", "us 500", "sandp"],
    "nasdaq100": ["nasdaq", "nasdaq 100", "ndx", "qqq"],
    "nikkei225": ["nikkei", "nikkei 225", "jp225", "tse 225"],
}

UNIVERSES = {
    "dax": (DAX40, "DAX 40 (Xetra)"),
    "ftse100": (FTSE100, "FTSE 100 (LSE)"),
    "cac40": (CAC40, "CAC 40 (Euronext Paris)"),
    "nifty50": (NIFTY50, "NIFTY 50 (NSE)"),
    "sp500": (SP500_SAMPLE, "S&P 500 (sample)"),
    "nasdaq100": (NASDAQ100, "NASDAQ-100"),
    "nikkei225": (NIKKEI225, "Nikkei 225 (TSE)"),
}

def resolve_universe(query: str) -> Tuple[str, List[str], str]:
    """
    Return (key, tickers, human_label) for a user query mentioning an index/exchange.
    If no match, returns ("", [], "").
    """
    q = (query or "").lower()
    # exact alias hit
    for key, phrs in ALIASES.items():
        for p in phrs:
            if re.search(rf"\b{re.escape(p)}\b", q):
                tickers, label = UNIVERSES[key]
                return key, tickers, label
    # fallback: detect raw keywords
    if "dax" in q: return "dax", UNIVERSES["dax"][0], UNIVERSES["dax"][1]
    if "ftse" in q: return "ftse100", UNIVERSES["ftse100"][0], UNIVERSES["ftse100"][1]
    if "cac" in q: return "cac40", UNIVERSES["cac40"][0], UNIVERSES["cac40"][1]
    if "nifty" in q or "nse" in q: return "nifty50", UNIVERSES["nifty50"][0], UNIVERSES["nifty50"][1]
    if "s&p" in q or "sp500" in q or "s and p" in q: return "sp500", UNIVERSES["sp500"][0], UNIVERSES["sp500"][1]
    if "nasdaq" in q: return "nasdaq100", UNIVERSES["nasdaq100"][0], UNIVERSES["nasdaq100"][1]
    if "nikkei" in q or "jp225" in q: return "nikkei225", UNIVERSES["nikkei225"][0], UNIVERSES["nikkei225"][1]
    return "", [], ""
