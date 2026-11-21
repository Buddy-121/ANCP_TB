"""Data ingestion: file reading, header detection, date/period parsing, numeric cleaning,
automatic column mapping and conversion to the standard long format.

Nothing here assumes fixed column names. Every mapping is a *suggestion* that the
Data Management page shows to the user for confirmation or override.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
QUARTER_END_MONTH = {1: 6, 2: 9, 3: 12, 4: 3}  # Indian fiscal quarters (Q1 = Apr–Jun)
NA_TOKENS = {"", "-", "--", "—", "–", "na", "n/a", "nan", "none", "null", ".", "nm", "#n/a", "#value!"}


# --------------------------------------------------------------------------- reading
@dataclass
class RawTable:
    name: str
    df: pd.DataFrame                    # data with detected header
    preamble: list[str] = field(default_factory=list)  # text above the header (titles, units)
    header_row: int = 0


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", str(s).lower()).strip()


def read_any(file_bytes: bytes, filename: str) -> dict[str, pd.DataFrame]:
    """Return {sheet_name: raw DataFrame without header} for CSV/Excel."""
    name = filename.lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        engine = "xlrd" if name.endswith(".xls") else "openpyxl"
        sheets = pd.read_excel(bio, sheet_name=None, header=None, dtype=object, engine=engine)
        return {k: v for k, v in sheets.items() if not v.dropna(how="all").empty}
    # CSV / TXT: try encodings & separators
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sep = max([",", ";", "\t", "|"], key=lambda s: text[:5000].count(s))
    rows = pd.read_csv(io.StringIO(text), header=None, dtype=object, sep=sep,
                       skip_blank_lines=False, engine="python", on_bad_lines="skip")
    return {"CSV": rows}


def detect_header(raw: pd.DataFrame, max_scan: int = 25, header_override: int | None = None) -> RawTable:
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    scan = raw.head(max_scan)
    counts = scan.notna().sum(axis=1)
    max_count = counts.max() if len(counts) else 0
    header_idx = 0
    for i, c in counts.items():
        if c >= max(2, 0.6 * max_count):
            header_idx = i
            break
    if header_override is not None and 0 <= header_override < len(raw):
        header_idx = header_override
    preamble = [" ".join(str(x) for x in raw.iloc[i].dropna().tolist()) for i in range(header_idx)]
    header = raw.iloc[header_idx].tolist()
    cols, seen = [], {}
    for j, h in enumerate(header):
        if h is None or (isinstance(h, float) and np.isnan(h)) or str(h).strip() == "":
            name = f"col_{j}"
        elif isinstance(h, (pd.Timestamp, np.datetime64)) or hasattr(h, "strftime"):
            name = pd.Timestamp(h).strftime("%Y-%m-%d")
        else:
            name = str(h).strip()
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        cols.append(name)
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = cols
    df = df.dropna(how="all").reset_index(drop=True)
    return RawTable(name="", df=df, preamble=preamble, header_row=header_idx)


# --------------------------------------------------------------------------- cleaning
def to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    def conv(x):
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        t = str(x).strip()
        if t.lower() in NA_TOKENS:
            return np.nan
        neg = t.startswith("(") and t.endswith(")")
        t = re.sub(r"(₹|rs\.?|inr|usd|\$|%|,|\s|x$)", "", t, flags=re.I).strip("()")
        try:
            v = float(t)
        except ValueError:
            return np.nan
        return -v if neg else v

    return s.map(conv).astype(float)


def _fy_year(y: str) -> int:
    y = int(y)
    return y + 2000 if y < 100 else y


def parse_period_value(v) -> pd.Timestamp | None:
    """Parse a single period label (statement headers, brokerage periods).

    Returns the period END date. Handles: datetimes, 'Mar 2024', 'Mar-24', "Mar '24",
    'FY24', 'FY2024', '2023-24', '2024' (treated as fiscal year ending March),
    'Q1FY25', 'Q1 FY 25', 'FY25 Q1', '1QFY25', 'Jun-24', Excel serial dates.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "strftime"):
        return pd.Timestamp(v)
    if isinstance(v, (int, float, np.number)):
        x = float(v)
        if 1950 <= x <= 2100 and x == int(x):
            return pd.Timestamp(int(x), 3, 31)
        if 20000 < x < 80000:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(x))
        return None
    s = str(v).strip()
    if not s:
        return None
    sl = s.lower().replace("’", "'")
    # Quarter formats
    m = (re.match(r"^q([1-4])\s*[-' ]?\s*fy\s*'?(\d{2,4})$", sl) or
         re.match(r"^([1-4])q\s*fy\s*'?(\d{2,4})$", sl))
    if m:
        q, fy = int(m.group(1)), _fy_year(m.group(2))
        mon = QUARTER_END_MONTH[q]
        year = fy if q == 4 else fy - 1
        return pd.Timestamp(year, mon, 1) + pd.offsets.MonthEnd(0)
    m = re.match(r"^fy\s*'?(\d{2,4})\s*[-_ ]?\s*q([1-4])$", sl)
    if m:
        fy, q = _fy_year(m.group(1)), int(m.group(2))
        year = fy if q == 4 else fy - 1
        return pd.Timestamp(year, QUARTER_END_MONTH[q], 1) + pd.offsets.MonthEnd(0)
    m = re.match(r"^fy\s*'?(\d{2,4})$", sl)  # 'FY 2027 Est' (estimates) is deliberately NOT matched
    if m:
        return pd.Timestamp(_fy_year(m.group(1)), 3, 31)
    m = re.match(r"^(\d{4})\s*[-/]\s*(\d{2}|\d{4})$", sl)
    if m and int(m.group(1)) > 1950:  # fiscal year label like 2023-24 / 2023-2024 / 1999-00
        start = int(m.group(1))
        end = int(m.group(2)) if len(m.group(2)) == 4 else (start // 100) * 100 + int(m.group(2))
        if end < start:
            end += 100
        if end == start + 1:
            return pd.Timestamp(end, 3, 31)
    m = re.match(r"^(\d{4})$", sl)
    if m and 1950 <= int(sl) <= 2100:
        return pd.Timestamp(int(sl), 3, 31)
    m = re.match(r"^([a-z]{3})[a-z]*\s*[-' /.]*\s*'?(\d{2}|\d{4})$", sl)
    if m and m.group(1) in MONTHS:
        year = _fy_year(m.group(2))
        return pd.Timestamp(year, MONTHS[m.group(1)], 1) + pd.offsets.MonthEnd(0)
    if sl in {"ttm", "ltm", "latest"}:
        return None
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}", s):
            return pd.Timestamp(s[:10])
        return pd.to_datetime(s, dayfirst=True, errors="raise")
    except (ValueError, TypeError, OverflowError):
        return None


def parse_date_series(s: pd.Series) -> tuple[pd.Series, dict]:
    """Parse a date column. Resolves day/month ambiguity from the data itself."""
    info = {"dayfirst": None, "ambiguous": False, "unparsed": 0}
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s), info
    sample = s.dropna().astype(str).str.strip()
    nums = sample.str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")
    dayfirst = True
    if nums.notna().all(axis=1).any():
        a = pd.to_numeric(nums[0], errors="coerce")
        b = pd.to_numeric(nums[1], errors="coerce")
        if (a > 12).any():
            dayfirst = True
        elif (b > 12).any():
            dayfirst = False
        else:
            info["ambiguous"] = True  # assume day-first (Indian convention) – flagged
        info["dayfirst"] = dayfirst

    def one(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return pd.NaT
        if hasattr(x, "strftime"):
            return pd.Timestamp(x)
        t = str(x).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}", t):
            return pd.to_datetime(t[:10], errors="coerce")
        if re.match(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$", t):
            return pd.to_datetime(t, dayfirst=dayfirst, errors="coerce")
        p = parse_period_value(x)
        return pd.NaT if p is None else p

    out = s.map(one)
    out = pd.to_datetime(out, errors="coerce")
    info["unparsed"] = int(out.isna().sum() - s.isna().sum())
    return out, info


def date_parse_rate(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return 0.0
    sample = s.head(60)
    if pd.api.types.is_numeric_dtype(sample):
        return 0.0
    parsed, _ = parse_date_series(sample)
    return float(parsed.notna().mean())


def numeric_rate(s: pd.Series) -> float:
    s = s.dropna()
    return 0.0 if s.empty else float(to_number(s.head(200)).notna().mean())


# --------------------------------------------------------------------------- mapping
def _match(cols: list[str], exact: list[str], contains: list[str] | None = None,
           exclude: list[str] | None = None, used: set | None = None) -> str | None:
    used = used or set()
    exclude = exclude or []
    normed = {c: _norm(c) for c in cols if c not in used}
    for cand in exact:
        for c, n in normed.items():
            if n == cand:
                return c
    for cand in contains or []:
        for c, n in normed.items():
            if cand in n and not any(e in n for e in exclude):
                return c
    return None


def best_date_col(df: pd.DataFrame) -> str | None:
    rates = {c: date_parse_rate(df[c]) for c in df.columns}
    named = [c for c in df.columns if any(k in _norm(c) for k in ("date", "period", "time", "month", "year", "quarter"))]
    for c in named:
        if rates.get(c, 0) > 0.8:
            return c
    best = max(rates, key=rates.get) if rates else None
    return best if best and rates[best] > 0.8 else None


def unit_hint(table: RawTable) -> str | None:
    text = " ".join(table.preamble + [str(c) for c in table.df.columns]).lower()
    if not table.df.empty:
        text += " " + " ".join(table.df.iloc[:, 0].astype(str).head(80).tolist()).lower()
    if re.search(r"\b(crore|crores|cr\.?|rs\.? ?cr)\b", text):
        return "₹ Crore"
    if re.search(r"\b(lakh|lakhs|lac|lacs)\b", text):
        return "₹ Lakh"
    if re.search(r"\b(million|mn|mln)\b", text):
        return "₹ Million"
    if re.search(r"\b(billion|bn)\b", text):
        return "₹ Billion"
    return None


EXP_KW = ["expected", "expectation", "expectations", "estimate", "estimates", "est", "consensus",
          "forecast", "street", "brokerage", "exp", "estimated", "poll"]
ACT_KW = ["actual", "actuals", "reported", "act", "results"]


def _classify_ea(col: str) -> tuple[str | None, str]:
    toks = _norm(col).split()
    kind = None
    if any(t in EXP_KW for t in toks):
        kind = "expected"
    elif any(t in ACT_KW for t in toks):
        kind = "actual"
    base = " ".join(t for t in toks if t not in EXP_KW + ACT_KW + ["vs", "v", "of", "the"])
    return kind, base.strip()


def suggest_mapping(table: RawTable, kind: str) -> dict:
    df = table.df
    cols = list(df.columns)
    if kind == "timeseries":
        date_col = best_date_col(df)
        num_cols = [c for c in cols if c != date_col and numeric_rate(df[c]) > 0.7]
        value_col = _match(num_cols, ["close", "price", "value", "rate", "adj close"],
                           ["close", "price", "rate", "value", "last", "usd", "inr", "brent", "wti", "dcoil"],
                           exclude=["prev", "open"]) or (num_cols[0] if num_cols else None)
        return {"date_col": date_col, "value_col": value_col}

    if kind == "ohlc":
        date_col = best_date_col(df)
        used = {date_col}
        m = {"date_col": date_col}
        for key, exact, contains, excl in [
            ("open_col", ["open", "open price"], ["open"], ["interest"]),
            ("high_col", ["high", "high price"], ["high"], ["52"]),
            ("low_col", ["low", "low price"], ["low"], ["52"]),
            ("close_col", ["close", "close price", "closing price"], ["close", "ltp", "last"], ["prev", "adj"]),
            ("volume_col", ["volume", "total traded quantity", "shares traded", "no of shares"],
             ["volume", "traded quantity", "qty", "shares"], ["value", "turnover"]),
        ]:
            c = _match(cols, exact, contains, excl, used)
            m[key] = c
            if c:
                used.add(c)
        return m

    if kind == "statement":
        period_cols = [c for c in cols[1:] if parse_period_value(c) is not None]
        if len(period_cols) >= 2:
            text_rates = {c: df[c].dropna().map(lambda x: isinstance(x, str)).mean() if df[c].notna().any() else 0
                          for c in cols if c not in period_cols}
            item_col = max(text_rates, key=text_rates.get) if text_rates else cols[0]
            return {"orientation": "items_in_rows", "item_col": item_col, "period_cols": period_cols}
        date_col = best_date_col(df)
        item_cols = [c for c in cols if c != date_col and numeric_rate(df[c]) > 0.5]
        return {"orientation": "periods_in_rows", "period_col": date_col, "item_cols": item_cols}

    if kind == "brokerage":
        work = df
        period_cols = [c for c in cols[1:] if parse_period_value(c) is not None]
        transposed = False
        if len(period_cols) >= 2:
            work = brokerage_transpose(df, cols[0], period_cols)
            transposed = True
        wcols = list(work.columns)
        period_col = "Period" if transposed else best_date_col(work)
        if period_col is None:
            # periods like Q1FY25 may not parse as dates in best_date_col – fall back
            rates = {c: work[c].dropna().head(40).map(lambda x: parse_period_value(x) is not None).mean()
                     for c in wcols if work[c].notna().any()}
            period_col = max(rates, key=rates.get) if rates and max(rates.values()) > 0.7 else None
        classified = {c: _classify_ea(c) for c in wcols if c != period_col}
        exp_cols = [c for c, (k, _) in classified.items() if k == "expected"]
        act_cols = [c for c, (k, _) in classified.items() if k == "actual"]
        if len(exp_cols) == 1 and len(act_cols) == 1:
            others = [c for c in wcols if c not in (period_col, exp_cols[0], act_cols[0])
                      and work[c].dropna().map(lambda x: isinstance(x, str)).mean() > 0.7]
            if others:
                return {"format": "long", "transposed": transposed, "period_col": period_col,
                        "metric_col": others[0], "expected_col": exp_cols[0], "actual_col": act_cols[0]}
        pairs = []
        for e in exp_cols:
            base = classified[e][1]
            match = next((a for a in act_cols if classified[a][1] == base), None)
            if match:
                words = [w for w in re.split(r"[\s_\-:()]+", str(e)) if w and w.lower() not in EXP_KW + ["vs", "v"]]
                pairs.append({"metric": " ".join(words) or e, "expected_col": e, "actual_col": match})
        return {"format": "wide", "transposed": transposed, "period_col": period_col, "pairs": pairs}
    raise ValueError(kind)


def brokerage_transpose(df: pd.DataFrame, item_col: str, period_cols: list[str]) -> pd.DataFrame:
    w = df[[item_col] + period_cols].dropna(subset=[item_col]).copy()
    w[item_col] = w[item_col].astype(str).str.strip()
    w = w.drop_duplicates(subset=[item_col]).set_index(item_col).T
    w.index.name = "Period"
    return w.reset_index()


# --------------------------------------------------------------------------- transform
def _dedupe_items(items: pd.Series) -> pd.Series:
    seen: dict[str, int] = {}
    out = []
    for it in items:
        it = re.sub(r"\s+", " ", str(it)).strip()
        if it in seen:
            seen[it] += 1
            out.append(f"{it} ({seen[it]})")
        else:
            seen[it] = 1
            out.append(it)
    return pd.Series(out, index=items.index)


def transform(table: RawTable, kind: str, mapping: dict, multiplier: float | None = None
              ) -> tuple[pd.DataFrame, dict]:
    """Convert to long format [period, item, value, value2]. Returns (long_df, stats)."""
    df = table.df
    stats: dict = {}
    if kind == "timeseries":
        dates, info = parse_date_series(df[mapping["date_col"]])
        vals = to_number(df[mapping["value_col"]])
        out = pd.DataFrame({"period": dates, "item": "value", "value": vals})
        stats.update(date_info=info, non_numeric=int(vals.isna().sum() - df[mapping["value_col"]].isna().sum()))
    elif kind == "ohlc":
        dates, info = parse_date_series(df[mapping["date_col"]])
        parts = []
        for f in ("open", "high", "low", "close", "volume"):
            c = mapping.get(f"{f}_col")
            if c:
                parts.append(pd.DataFrame({"period": dates, "item": f, "value": to_number(df[c])}))
        out = pd.concat(parts, ignore_index=True)
        stats.update(date_info=info)
    elif kind == "statement":
        if mapping["orientation"] == "items_in_rows":
            items = _dedupe_items(df[mapping["item_col"]].fillna("").astype(str))
            parts = []
            for pc in mapping["period_cols"]:
                p = parse_period_value(pc)
                parts.append(pd.DataFrame({"period": p, "item": items, "value": to_number(df[pc])}))
            out = pd.concat(parts, ignore_index=True)
            out = out[out["item"].str.strip() != ""]
        else:
            dates, info = parse_date_series(df[mapping["period_col"]])
            parts = [pd.DataFrame({"period": dates, "item": str(c).strip(), "value": to_number(df[c])})
                     for c in mapping["item_cols"]]
            out = pd.concat(parts, ignore_index=True)
            stats.update(date_info=info)
        if multiplier is not None:
            out["value"] = out["value"] * multiplier
    elif kind == "brokerage":
        work = df
        if mapping.get("transposed"):
            period_cols = [c for c in df.columns[1:] if parse_period_value(c) is not None]
            work = brokerage_transpose(df, df.columns[0], period_cols)
        periods = work[mapping["period_col"]].map(parse_period_value)
        periods = pd.to_datetime(periods, errors="coerce")
        if mapping["format"] == "long":
            out = pd.DataFrame({"period": periods, "item": work[mapping["metric_col"]].astype(str).str.strip(),
                                "value": to_number(work[mapping["expected_col"]]),
                                "value2": to_number(work[mapping["actual_col"]])})
        else:
            parts = [pd.DataFrame({"period": periods, "item": p["metric"],
                                   "value": to_number(work[p["expected_col"]]),
                                   "value2": to_number(work[p["actual_col"]])}) for p in mapping["pairs"]]
            out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
                columns=["period", "item", "value", "value2"])
    else:
        raise ValueError(kind)

    if "value2" not in out:
        out["value2"] = np.nan
    stats["rows_in"] = len(df)
    stats["unparsed_periods"] = int(out["period"].isna().sum())
    out = out.dropna(subset=["period"])
    value_cols = ["value", "value2"] if kind == "brokerage" else ["value"]
    stats["missing_values"] = int(out[value_cols].isna().all(axis=1).sum())
    stats["duplicates"] = int(out.duplicated(subset=["period", "item"]).sum())
    out = out.dropna(subset=value_cols, how="all")
    out = out.drop_duplicates(subset=["period", "item"], keep="last")
    out = out.sort_values(["item", "period"]).reset_index(drop=True)
    stats["rows_out"] = len(out)
    return out, stats


# --------------------------------------------------------------------------- merge
def merge_incremental(existing: pd.DataFrame, new: pd.DataFrame, overwrite_conflicts: bool
                      ) -> tuple[pd.DataFrame, dict]:
    """Merge new records into existing history without silently overwriting.

    Rows for new (period, item) keys are appended. For keys present in both with
    different values, existing values are kept unless overwrite_conflicts=True.
    """
    key = ["period", "item"]
    if existing.empty:
        return new.copy(), {"added": len(new), "conflicts": 0, "unchanged": 0}
    e = existing.set_index(key)
    n = new.set_index(key)
    common = e.index.intersection(n.index)
    diff_mask = ~np.isclose(e.loc[common, "value"].astype(float), n.loc[common, "value"].astype(float),
                            rtol=1e-6, equal_nan=True)
    conflicts = common[diff_mask]
    added = n.index.difference(e.index)
    merged = e.copy()
    if overwrite_conflicts and len(conflicts):
        merged.loc[conflicts, ["value", "value2"]] = n.loc[conflicts, ["value", "value2"]].values
    merged = pd.concat([merged, n.loc[added]])
    merged = merged.reset_index().sort_values(key).reset_index(drop=True)
    conflict_df = pd.DataFrame({"existing": e.loc[conflicts, "value"], "new": n.loc[conflicts, "value"]}).reset_index()
    return merged, {"added": len(added), "conflicts": len(conflicts), "unchanged": len(common) - len(conflicts),
                    "conflict_rows": conflict_df}
