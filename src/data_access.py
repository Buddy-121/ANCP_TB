"""Read helpers used by analytics and pages: current datasets in convenient shapes,
plus key line-item resolution (auto-match by name, user overrides take priority)."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from . import database as db
from .config import DATASETS

KEY_ITEMS: dict[str, dict[str, dict]] = {
    "income": {
        "revenue": {"p": ["revenue from operations", "revenue", "total revenue", "net sales", "revenue from operations net", "total revenue from operations",
                          "net revenue", "sales", "revenue", "total revenue", "income from operations"],
                    "x": ["other", "cost", "growth"]},
        "total_income": {"p": ["total income", "total revenue"], "x": ["other", "comprehensive"]},
        "other_income": {"p": ["other income"], "x": ["comprehensive"]},
        "raw_material": {"p": ["raw materials consumables used", "cost of materials consumed", "cost of raw materials consumed", "raw material cost",
                               "raw materials consumed", "cost of materials", "raw material", "material cost",
                               "cost of goods sold", "cogs"], "x": []},
        "purchases": {"p": ["purchases of merchandise", "purchases of stock in trade", "purchase of stock in trade", "purchases of traded goods",
                            "purchase of traded goods"], "x": []},
        "inventory_change": {"p": ["changes in inventories of finished goods work in progress and stock in trade",
                                   "changes in inventories", "change in inventories", "increase decrease in inventories"],
                             "x": []},
        "employee_cost": {"p": ["personnel expenses", "employee benefits expense", "employee benefit expenses", "employee benefit expense",
                                "employee cost", "staff cost", "employee expenses"], "x": []},
        "other_expenses": {"p": ["other expenses"], "x": []},
        "ebitda": {"p": ["ebitda", "operating profit", "pbdit", "profit before depreciation interest and tax"],
                   "x": ["margin", "%"]},
        "depreciation": {"p": ["depreciation expense", "depreciation amortization", "depreciation and amortisation expense", "depreciation and amortization expense",
                               "depreciation and amortisation", "depreciation"], "x": []},
        "finance_cost": {"p": ["finance costs", "finance cost", "interest expense", "interest"],
                         "x": ["income", "received"]},
        "pbt": {"p": ["pretax income loss gaap", "pre tax income", "profit before tax", "profit loss before tax", "pbt", "profit before exceptional items and tax",
                      "profit loss before exceptional items and tax"], "x": ["margin"]},
        "tax": {"p": ["income tax expense benefit", "income tax expense", "total tax expense", "tax expense", "total tax", "tax"],
                "x": ["before", "profit", "deferred", "current", "rate"]},
        "net_profit": {"p": ["net income gaap", "profit for the year", "profit for the period", "profit loss for the period",
                             "profit loss for the year", "net profit loss for the period", "net profit", "profit after tax",
                             "pat", "net income", "profit attributable to owners", "profit loss after tax"],
                       "x": ["margin", "before", "growth", "comprehensive"]},
        "eps": {"p": ["basic eps gaap", "adjusted eps", "basic eps", "eps basic", "earnings per share basic", "basic", "eps"], "x": ["diluted"]},
    },
    "balance": {
        "total_assets": {"p": ["total assets"], "x": ["current", "non current"]},
        "total_equity": {"p": ["total equity", "shareholders funds", "total shareholders funds", "net worth",
                               "equity attributable to owners of the company", "total shareholders equity"],
                         "x": ["liabilities"]},
        "share_capital": {"p": ["equity share capital", "common stock", "share capital"], "x": ["apic"]},
        "lt_borrowings": {"p": ["lt debt", "long term borrowings", "non current borrowings", "borrowings non current"], "x": []},
        "st_borrowings": {"p": ["st debt", "short term borrowings", "current borrowings", "borrowings current"], "x": []},
        "total_debt": {"p": ["total debt", "total borrowings", "borrowings"],
                       "x": ["long", "short", "current", "non", "term", "st ", "lt "]},
        "current_assets": {"p": ["total current assets", "current assets"], "x": ["non"]},
        "current_liabilities": {"p": ["total current liabilities", "current liabilities"], "x": ["non"]},
        "inventory": {"p": ["inventories", "inventory"], "x": ["change"]},
        "receivables": {"p": ["accounts notes receiv", "accounts receivable net", "trade receivables", "sundry debtors", "debtors", "receivables"], "x": ["non current"]},
        "payables": {"p": ["accounts payable", "trade payables", "total trade payables", "sundry creditors", "creditors", "payables"],
                     "x": ["non current"]},
        "cash": {"p": ["cash cash equivalents", "cash and cash equivalents", "cash and bank balances", "cash and bank", "cash"], "x": ["flow"]},
        "current_investments": {"p": ["st investments", "current investments"], "x": ["non"]},
    },
    "cashflow": {
        "cfo": {"p": ["cash from operating activities", "net cash from operating activities", "net cash generated from operating activities",
                      "net cash flow from operating activities", "cash flow from operating activities",
                      "cash from operating activity", "operating cash flow", "net cash generated by operating activities"],
                "x": ["before"]},
        "cfi": {"p": ["cash from investing activities", "net cash used in investing activities", "net cash from investing activities",
                      "cash flow from investing activities", "cash from investing activity", "investing activities"],
                "x": []},
        "cff": {"p": ["cash from financing activities", "net cash used in financing activities", "net cash from financing activities",
                      "cash flow from financing activities", "cash from financing activity", "financing activities"],
                "x": []},
        "capex": {"p": ["acq of fixed prod assets", "acq of fixed intang", "purchase of property plant and equipment", "purchase of fixed assets", "capital expenditure",
                        "capex", "payment for property plant and equipment", "purchase of property plant equipment",
                        "acquisition of property plant and equipment", "fixed assets purchased"], "x": ["sale"]},
        "dividends": {"p": ["dividends paid", "dividend paid", "dividends paid to shareholders"], "x": ["received"]},
    },
}

KEY_ITEMS["income_q"] = KEY_ITEMS["income"]

KEY_LABELS = {
    "revenue": "Revenue from operations", "total_income": "Total income", "other_income": "Other income",
"raw_material": "Raw material cost",
    "purchases": "Purchases of traded goods", "inventory_change": "Change in inventories",
    "employee_cost": "Employee cost", "other_expenses": "Other expenses", "ebitda": "EBITDA (reported)",
    "depreciation": "Depreciation", "finance_cost": "Finance cost", "pbt": "Profit before tax", "tax": "Tax expense",
    "net_profit": "Net profit (PAT)", "eps": "EPS (basic)", "total_assets": "Total assets",
    "total_equity": "Total equity", "share_capital": "Equity share capital", "lt_borrowings": "Long-term borrowings",
    "st_borrowings": "Short-term borrowings", "total_debt": "Total debt", "current_assets": "Current assets",
    "current_liabilities": "Current liabilities", "inventory": "Inventories", "receivables": "Trade receivables",
    "payables": "Trade payables", "cash": "Cash & equivalents", "current_investments": "Current investments",
    "cfo": "Cash from operations", "cfi": "Cash from investing", "cff": "Cash from financing",
    "capex": "Capital expenditure", "dividends": "Dividends paid",
}


def _n(s: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", str(s).lower()).strip()


def current_id(category: str) -> int | None:
    m = db.current_meta(category)
    return m["id"] if m else None


def has(category: str) -> bool:
    return current_id(category) is not None


@st.cache_data(show_spinner=False)
def _records(ds_id: int) -> pd.DataFrame:
    return db.load_records(ds_id)


def records(category: str) -> pd.DataFrame:
    i = current_id(category)
    if i is None:
        return pd.DataFrame(columns=["period", "item", "value", "value2"])
    return _records(i).copy()


def price_series(category: str) -> pd.Series:
    """Daily close (OHLC) or value (single time series)."""
    r = records(category)
    field = "close" if DATASETS[category]["kind"] == "ohlc" else "value"
    s = r[r["item"] == field].set_index("period")["value"].sort_index()
    s.name = category
    return s[~s.index.duplicated(keep="last")]


def ohlc_frame() -> pd.DataFrame:
    r = records("ohlc")
    if r.empty:
        return pd.DataFrame()
    return r.pivot_table(index="period", columns="item", values="value").sort_index()


def statement_wide(category: str) -> pd.DataFrame:
    """Index = line item (original order preserved), columns = period (ascending)."""
    r = records(category)
    if r.empty:
        return pd.DataFrame()
    order = list(dict.fromkeys(r["item"]))
    w = r.pivot_table(index="item", columns="period", values="value", aggfunc="last")
    return w.reindex([i for i in order if i in w.index]).sort_index(axis=1)


def periodicity(columns) -> str:
    idx = pd.DatetimeIndex(sorted(columns))
    if len(idx) < 2:
        return "annual"
    gap = pd.Series(idx).diff().dt.days.median()
    return "quarterly" if gap < 200 else "annual"


def auto_match(items: list[str], spec: dict) -> str | None:
    normed = {i: _n(i) for i in items}
    excl = spec.get("x", [])
    for p in spec["p"]:
        for i, n in normed.items():
            if n == p:
                return i
    for p in spec["p"]:
        for i, n in normed.items():
            if n.startswith(p) and not any(x in n for x in excl):
                return i
    for p in spec["p"]:
        if len(p) < 5:
            continue
        for i, n in normed.items():
            if p in n and not any(x in n for x in excl):
                return i
    return None


def resolved_items(category: str) -> dict[str, str | None]:
    w = statement_wide(category)
    items = list(w.index)
    overrides = db.get_item_map(category)
    out = {}
    for key, spec in KEY_ITEMS.get(category, {}).items():
        ov = overrides.get(key)
        out[key] = ov if ov in items else auto_match(items, spec)
    return out


def key_series(category: str, key: str) -> pd.Series | None:
    w = statement_wide(category)
    if w.empty:
        return None
    item = resolved_items(category).get(key)
    if item is None:
        if category == "balance" and key == "total_debt":
            parts = [key_series("balance", k) for k in ("lt_borrowings", "st_borrowings")]
            parts = [p for p in parts if p is not None]
            if parts:
                return pd.concat(parts, axis=1).sum(axis=1, min_count=1).rename("total_debt")
        return None
    s = w.loc[item].astype(float)
    s.name = key
    return s


def brokerage(category: str) -> pd.DataFrame:
    r = records(category)
    if r.empty:
        return r
    r = r.rename(columns={"item": "metric", "value": "expected", "value2": "actual"})
    return r.sort_values(["metric", "period"]).reset_index(drop=True)


def data_status() -> pd.DataFrame:
    rows = []
    for cat, meta in DATASETS.items():
        m = db.current_meta(cat)
        rows.append({"Dataset": meta["label"], "Key": cat, "Loaded": bool(m),
                     "Version": m["version"] if m else None, "Source": m["source"] if m else None,
                     "From": m["period_start"] if m else None, "To": m["period_end"] if m else None,
                     "Records": m["row_count"] if m else 0, "Unit": m["unit"] if m else None,
                     "Updated": m["created_at"] if m else None})
    return pd.DataFrame(rows)


def clear_cache() -> None:
    _records.clear()
