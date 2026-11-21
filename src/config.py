"""Central configuration: paths, dataset catalogue, units, ticker symbols."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.getenv("APP_DB_PATH", DATA_DIR / "db" / "platform.sqlite"))
RAW_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

COMPANY = "Asian Paints Limited"
FISCAL_YEAR_END_MONTH = 3  # Indian fiscal year ends in March

# kind: timeseries (date + single value), ohlc (daily prices),
#       statement (line items x periods), brokerage (expected vs actual)
DATASETS: dict[str, dict] = {
    "usdinr": {"label": "USD/INR Exchange Rate", "kind": "timeseries", "ticker": "INR=X",
               "unit": "INR per USD"},
    "brent": {"label": "Brent Crude – Europe", "kind": "timeseries", "ticker": "BZ=F",
              "unit": "USD per barrel"},
    "brent_fut": {"label": "Brent Futures – ICE Dec-26 (COZ6)", "kind": "timeseries", "unit": "USD per barrel"},
    "wti": {"label": "WTI Crude – US", "kind": "timeseries", "ticker": "CL=F",
            "unit": "USD per barrel"},
    "ohlc": {"label": "Asian Paints Daily OHLC", "kind": "ohlc", "ticker": "ASIANPAINT.NS",
             "unit": "INR per share"},
    "income": {"label": "Income Statement (annual)", "kind": "statement"},
    "income_q": {"label": "Quarterly P&L (Bloomberg actuals)", "kind": "statement"},
    "balance": {"label": "Balance Sheet", "kind": "statement"},
    "cashflow": {"label": "Cash Flow Statement", "kind": "statement"},
    "ratios": {"label": "Financial Ratios", "kind": "statement"},
    "brokerage_q": {"label": "Brokerage Expectations vs Actuals – Quarterly", "kind": "brokerage"},
    "brokerage_y": {"label": "Brokerage Expectations vs Actuals – Yearly", "kind": "brokerage"},
    "consensus_range_q": {"label": "Consensus range (low/high) – Quarterly", "kind": "brokerage", "internal": True},
    "consensus_range_y": {"label": "Consensus range (low/high) – Yearly", "kind": "brokerage", "internal": True},
    "consensus_fy": {"label": "Bloomberg consensus estimates – forward FY", "kind": "statement", "internal": True},
}
UPLOADABLE = [k for k, v in DATASETS.items() if not v.get("internal")]

# Macro / commodity series shown across the app (only loaded ones are used)
MACRO_SERIES = ["brent", "brent_fut", "wti", "usdinr"]
SHORT_NAMES = {"brent": "Brent", "brent_fut": "Brent Fut (COZ6)", "wti": "WTI", "usdinr": "USD/INR",
               "ohlc": "Asian Paints"}

# Bloomberg workbooks bundled with the project (auto-loaded into an empty database)
INPUT_DIR = DATA_DIR / "input"

# Units: multiplier that converts the source unit into ₹ Crore (internal standard)
UNIT_TO_CRORE = {
    "₹ Crore": 1.0,
    "₹ Lakh": 0.01,
    "₹ Million": 0.1,
    "₹ Billion": 100.0,
    "₹ (absolute rupees)": 1e-7,
    "As reported (no conversion – ratios, %, per-share)": None,
}

PEERS = {
    "Asian Paints": "ASIANPAINT.NS",
    "Berger Paints": "BERGEPAINT.NS",
    "Kansai Nerolac": "KANSAINER.NS",
    "Indigo Paints": "INDIGOPNTS.NS",
    "Akzo Nobel India": "AKZOINDIA.NS",
}
BENCHMARK_INDEX = "^NSEI"

TRADING_DAYS = 252
