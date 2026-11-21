"""Data validation: produces a check list (PASS / WARN / FAIL) for every dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _row(check: str, status: str, detail: str) -> dict:
    return {"Check": check, "Status": status, "Detail": detail}


def validate(raw: pd.DataFrame, long_df: pd.DataFrame, kind: str, stats: dict) -> pd.DataFrame:
    out = []
    out.append(_row("Rows read", "PASS" if len(raw) else "FAIL", f"{len(raw):,} raw rows → {len(long_df):,} records"))

    miss = raw.isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        top = ", ".join(f"{k}: {v}" for k, v in miss.sort_values(ascending=False).head(6).items())
        out.append(_row("Missing cells (raw)", "WARN", top))
    else:
        out.append(_row("Missing cells (raw)", "PASS", "None"))

    dup_raw = int(raw.duplicated().sum())
    out.append(_row("Duplicate raw rows", "WARN" if dup_raw else "PASS", f"{dup_raw} fully duplicated rows"))

    up = stats.get("unparsed_periods", 0)
    out.append(_row("Unparseable dates/periods", "WARN" if up else "PASS", f"{up} records dropped"))
    di = stats.get("date_info") or {}
    if di.get("ambiguous"):
        out.append(_row("Date format ambiguity", "WARN",
                        "All day & month values ≤ 12 – assumed DD/MM (Indian convention). Verify a few dates."))
    dups = stats.get("duplicates", 0)
    out.append(_row("Duplicate period/item keys", "WARN" if dups else "PASS",
                    f"{dups} duplicates (last occurrence kept)"))
    mv = stats.get("missing_values", 0)
    out.append(_row("Non-numeric / empty values", "WARN" if mv else "PASS", f"{mv} records without a numeric value dropped"))

    if long_df.empty:
        out.append(_row("Usable data", "FAIL", "No usable records after mapping – check column mapping"))
        return pd.DataFrame(out)

    out.append(_row("Coverage", "PASS",
                    f"{long_df['period'].min():%d-%b-%Y} → {long_df['period'].max():%d-%b-%Y}, "
                    f"{long_df['period'].nunique():,} periods, {long_df['item'].nunique():,} items"))

    if kind in ("timeseries", "ohlc"):
        field = "value" if kind == "timeseries" else "close"
        s = long_df[long_df["item"] == field].set_index("period")["value"].sort_index()
        if len(s) > 2:
            gaps = s.index.to_series().diff().dt.days
            big = gaps[gaps > 7]
            out.append(_row("Date gaps > 7 days", "WARN" if len(big) else "PASS",
                            f"{len(big)} gaps; largest {int(gaps.max())} days" if len(big) else "None"))
            neg = int((s <= 0).sum())
            out.append(_row("Non-positive prices", "WARN" if neg else "PASS", f"{neg} values ≤ 0"))
            r = np.log(s[s > 0]).diff().dropna()
            if len(r) > 20:
                z = (r - r.mean()) / r.std()
                ext = z[abs(z) > 6]
                out.append(_row("Extreme daily moves (|z|>6)", "WARN" if len(ext) else "PASS",
                                ", ".join(f"{d:%d-%b-%Y}" for d in ext.index[:5]) or "None"))
        if kind == "ohlc":
            w = long_df.pivot_table(index="period", columns="item", values="value")
            if {"open", "high", "low", "close"} <= set(w.columns):
                bad = w[(w["high"] < w[["open", "close"]].max(axis=1) - 1e-9) |
                        (w["low"] > w[["open", "close"]].min(axis=1) + 1e-9)]
                out.append(_row("OHLC consistency", "WARN" if len(bad) else "PASS",
                                f"{len(bad)} rows where High/Low do not bound Open/Close"))
            else:
                out.append(_row("OHLC consistency", "WARN", "Open/High/Low/Close not all mapped"))

    if kind == "statement":
        w = long_df.pivot_table(index="item", columns="period", values="value")
        sparse = w.isna().mean(axis=1)
        out.append(_row("Sparse line items (>50% empty)", "WARN" if (sparse > 0.5).any() else "PASS",
                        f"{int((sparse > 0.5).sum())} of {len(w)} items"))
        items = {i.lower(): i for i in w.index}
        ta = next((items[k] for k in items if k == "total assets"), None) or \
            next((items[k] for k in items if k.startswith("total assets")), None)
        tl = next((items[k] for k in items if "total equity and liabilities" in k or k.startswith("total liabilities and")
                   or k.startswith("total liabilities & equity")), None)
        if ta and tl:
            d = (w.loc[ta] - w.loc[tl]).abs() / w.loc[ta].abs()
            out.append(_row("Balance sheet balances", "PASS" if (d.dropna() < 0.005).all() else "WARN",
                            f"Max difference {d.max():.2%} between '{ta}' and '{tl}'"))

    if kind == "brokerage":
        both = long_df.dropna(subset=["value", "value2"])
        out.append(_row("Expected & actual both present", "PASS" if len(both) == len(long_df) else "WARN",
                        f"{len(both)} of {len(long_df)} records have both values"))
    return pd.DataFrame(out)


def overall_status(checks: pd.DataFrame) -> str:
    if (checks["Status"] == "FAIL").any():
        return "FAIL"
    if (checks["Status"] == "WARN").any():
        return "WARN"
    return "PASS"
