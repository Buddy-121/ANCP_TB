"""Market data fetching via yfinance (Yahoo Finance – free, unofficial, no API key).

Limitations: Yahoo data is not an official exchange feed; futures tickers (BZ=F, CL=F)
are front-month continuous contracts, not spot prices; INR=X is an indicative rate.
"""
from __future__ import annotations

import pandas as pd


def fetch_history(ticker: str, start=None, end=None, period: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    kwargs = {"auto_adjust": False, "progress": False, "threads": False}
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end
    df = yf.download(ticker, **kwargs)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the ticker or your internet connection.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    df["period"] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
    return df


def to_long(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "timeseries":
        return pd.DataFrame({"period": df["period"], "item": "value", "value": df["close"].astype(float)}).dropna()
    parts = [pd.DataFrame({"period": df["period"], "item": f, "value": df[f].astype(float)})
             for f in ("open", "high", "low", "close", "volume") if f in df.columns]
    return pd.concat(parts, ignore_index=True).dropna()


def fetch_close_panel(tickers: dict[str, str], period: str = "5y") -> pd.DataFrame:
    """Close prices for several tickers, columns = names."""
    out = {}
    errors = {}
    for name, t in tickers.items():
        try:
            out[name] = fetch_history(t, period=period).set_index("period")["close"]
        except Exception as e:  # noqa: BLE001 - surface per-ticker errors to UI
            errors[name] = str(e)
    panel = pd.DataFrame(out).sort_index()
    panel.attrs["errors"] = errors
    return panel


def fetch_fundamentals(tickers: dict[str, str]) -> pd.DataFrame:
    """Snapshot fundamentals from Yahoo `info` (may be incomplete for Indian stocks)."""
    import yfinance as yf

    fields = {"marketCap": "Market Cap (₹ Cr)", "trailingPE": "P/E (TTM)", "priceToBook": "P/B",
              "returnOnEquity": "ROE", "profitMargins": "Net Margin", "operatingMargins": "Operating Margin",
              "revenueGrowth": "Revenue Growth (YoY)", "debtToEquity": "Debt/Equity (%)",
              "dividendYield": "Dividend Yield"}
    rows = []
    for name, t in tickers.items():
        row = {"Company": name, "Ticker": t}
        try:
            info = yf.Ticker(t).info or {}
        except Exception:  # noqa: BLE001
            info = {}
        for k, label in fields.items():
            v = info.get(k)
            if k == "marketCap" and v:
                v = v / 1e7
            row[label] = v
        rows.append(row)
    return pd.DataFrame(rows)
