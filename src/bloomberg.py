"""Importer for the Bloomberg workbooks used in this project.

Workbook 1 (e.g. CP_DATA1.xlsx) – Bloomberg terminal exports
  * Daily price sheets:   "Security / Start Date / ..." preamble, then
                          Date | PX_LAST | Change | % Change | PX_OPEN | ... | PX_HIGH | ... | PX_LOW | ...
  * Statement sheets:     "In Millions of INR except Per Share" | FY 2005 | FY 2006 | ... | Last 12M | FY 2027 Est
                          col A = line item, col B = Bloomberg field code; BS has "Growth (YoY/Seq)" rows.

Workbook 2 (e.g. CPD.xlsx) – Bloomberg BEst "Company Financial (Single Period)", one sheet per period
  * Row 2: "APNT IN Equity    Period:2018:Q1    Currency:INR"
  * Row 3: Metric | | | | Actuals | Mean Consensus | Low | High | Median | broker 1 | broker 2 ...

Units: Bloomberg reports INR in MILLIONS. Monetary values are converted to ₹ Crore (× 0.1).
Per-share values, percentages, ratios, day counts and share counts are NOT converted.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .ingestion import QUARTER_END_MONTH, to_number

MILLION_TO_CRORE = 0.1

NON_MONETARY = re.compile(
    r"(eps|per (basic |diluted )?share|/share|\bshares\b|dps|dividends per share|margin|%|ratio|\brate\b|percent|employees|"
    r"shareholders|\bdays\b|cycle|turnover|par value|price to|ebit/interest|interest expense to|t12m|"
    r"to equity|to net income|sales to|accounting standard|growth)", re.I)
# field-code patterns that are unambiguously non-monetary (names cover the rest)
NON_MONETARY_CODE = re.compile(r"(_PCT|MARGIN|^IS_EPS|_EPS_|EPS_ADJ|DILUTED_EPS|SH_OUT|PER_SH|PAR_VAL|_DPS|DIV_PER|"
                               r"NUM_OF_|AVG_NUM_SH|SH_FOR_DILUTED|EFF_TAX_RATE)")


def is_monetary(name: str, code: str | None = "") -> bool:
    return not (NON_MONETARY.search(str(name)) or NON_MONETARY_CODE.search(str(code or "")))


def _clean_item(s: str) -> str:
    s = re.sub(r"^\s*[+\-]\s+", "", str(s)).strip()
    return re.sub(r"\s+", " ", s)


# ============================================================ detection
def classify_sheet(raw: pd.DataFrame, sheet_name: str = "") -> tuple[str | None, str]:
    """Return (dataset key, description) for a CP_DATA-style sheet, or (None, reason)."""
    raw = raw.dropna(how="all", axis=1)
    if raw.empty:
        return None, "empty sheet"
    first = str(raw.iloc[0, 0]).strip()
    if first == "Security":
        sec = str(raw.iloc[0, 1]).upper()
        if "APNT" in sec:
            return "ohlc", f"Daily prices – {sec}"
        if "INR" in sec and "CURNCY" in sec:
            return "usdinr", f"Daily FX – {sec}"
        if "EUCRBRDT" in sec or sec.startswith("CO1"):
            return "brent", f"Brent spot – {sec}"
        if re.match(r"^CO[A-Z]\d", sec):
            return "brent_fut", f"ICE Brent futures contract – {sec}"
        if sec.startswith("CL") or "USCRWTIC" in sec:
            return "wti", f"WTI – {sec}"
        return None, f"Unrecognised security {sec}"
    text = " ".join(str(v) for v in raw.iloc[:6, 0].tolist())
    if "In Millions" in text or "12 Months Ending" in text:
        body = " ".join(str(v) for v in raw.iloc[5:12, 0].tolist()).lower()
        name = sheet_name.lower()
        if "total assets" in body or "cash, cash equivalents" in body or " bs" in f" {name}":
            return "balance", "Balance sheet (annual)"
        if "operating activities" in body or "cfs" in name or "cash flow" in name:
            return "cashflow", "Cash flow statement (annual)"
        if "revenue" in body or " is" in f" {name}":
            return "income", "Income statement (annual)"
    if "Company Financial" in first:
        return "consensus", "Bloomberg BEst consensus vs actuals"
    return None, "not a recognised Bloomberg layout"


# ============================================================ price sheets
def parse_prices(raw: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, dict]:
    raw = raw.reset_index(drop=True)
    meta = {}
    hdr = None
    for i in range(min(20, len(raw))):
        k = str(raw.iloc[i, 0]).strip()
        if k == "Date":
            hdr = i
            break
        if k and k != "nan" and pd.notna(raw.iloc[i, 1]):
            meta[k] = raw.iloc[i, 1]
    if hdr is None:
        raise ValueError("Could not find the 'Date' header row")
    cols = [str(c).strip() for c in raw.iloc[hdr].tolist()]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols if len(set(cols)) == len(cols) else [f"{c}__{j}" for j, c in enumerate(cols)]
    colmap = {c.split("__")[0]: c for c in reversed(df.columns)}  # first occurrence wins
    dates = pd.to_datetime(df[colmap["Date"]], errors="coerce")
    fields = {"PX_LAST": "close", "PX_OPEN": "open", "PX_HIGH": "high", "PX_LOW": "low", "PX_VOLUME": "volume"}
    parts = []
    for bbg, f in fields.items():
        if bbg in colmap:
            item = "value" if (kind == "timeseries" and f == "close") else f
            if kind == "timeseries" and f != "close":
                continue
            parts.append(pd.DataFrame({"period": dates, "item": item, "value": to_number(df[colmap[bbg]])}))
    out = pd.concat(parts, ignore_index=True).dropna(subset=["period", "value"])
    out = out.drop_duplicates(["period", "item"], keep="first").sort_values(["item", "period"]).reset_index(drop=True)
    out["value2"] = np.nan
    meta["fields"] = [f for f in fields if f in colmap]
    return out, meta


# ============================================================ annual statements
def parse_statement(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (actuals long df in ₹ Crore, estimates long df, meta)."""
    raw = raw.reset_index(drop=True)
    hdr = None
    for i in range(min(15, len(raw))):
        if any(re.match(r"^FY \d{4}", str(v)) for v in raw.iloc[i].tolist()):
            hdr = i
            break
    if hdr is None:
        raise ValueError("Could not find the FY header row")
    labels = raw.iloc[hdr].tolist()
    ends = raw.iloc[hdr + 1].tolist() if hdr + 1 < len(raw) else [None] * len(labels)
    unit_text = str(raw.iloc[hdr, 0])
    multiplier = MILLION_TO_CRORE if "million" in unit_text.lower() else 1.0
    act_cols, est_cols = {}, {}
    for j in range(2, len(labels)):
        lab = str(labels[j]).strip()
        end = pd.to_datetime(str(ends[j]), format="%m/%d/%Y", errors="coerce")
        if not re.match(r"^FY \d{4}", lab):
            continue  # skips 'Last 12M'
        if pd.isna(end):
            end = pd.Timestamp(int(lab[3:7]), 3, 31)
        (est_cols if "est" in lab.lower() else act_cols)[j] = end
    rows_a, rows_e = [], []
    seen: dict[str, int] = {}
    for i in range(hdr + 2, len(raw)):
        name = raw.iloc[i, 0]
        if pd.isna(name):
            continue
        sname = str(name)
        if "Growth (" in sname or sname.startswith("Source:") or "12 Months Ending" in sname:
            continue
        code = raw.iloc[i, 1] if raw.shape[1] > 1 else None
        vals = to_number(raw.iloc[i, list(act_cols)])
        evals = to_number(raw.iloc[i, list(est_cols)]) if est_cols else pd.Series(dtype=float)
        if vals.isna().all() and evals.isna().all():
            continue  # section headers / text rows
        item = _clean_item(sname)
        if item in seen:
            seen[item] += 1
            item = f"{item} ({seen[item]})"
        else:
            seen[item] = 1
        mult = multiplier if is_monetary(item, code) else 1.0
        for (j, end), v in zip(act_cols.items(), vals.values):
            if pd.notna(v):
                rows_a.append((end, item, v * mult))
        for (j, end), v in zip(est_cols.items(), evals.values):
            if pd.notna(v):
                rows_e.append((end, item, v * mult))
    mk = lambda r: pd.DataFrame(r, columns=["period", "item", "value"]).assign(value2=np.nan)  # noqa: E731
    meta = {"title": str(raw.iloc[1, 0]) if len(raw) > 1 else "", "unit_source": unit_text,
            "periods": len(act_cols), "estimate_periods": [d.strftime("%b-%Y") for d in est_cols.values()]}
    return mk(rows_a), mk(rows_e), meta


# ============================================================ BEst consensus workbook
# (field code, display name) – quarterly P&L actuals
PNL_FIELDS = [
    ("IS_COMP_SALES", "Revenue"),
    ("CB_IS_RAW_MATERIALS_AND_CONSUME_USED", "Raw Materials & Consumables Used"),
    ("CB_IS_PURCHASE_OF_MERCHANDISE", "Purchases Of Merchandise"),
    ("CB_IS_CHG_IN_INVENTORY_OF_FINISHED_GOOD_WIP", "Changes In Inventories Of Finished Goods and WiP"),
    ("IS_OPEX_COST_MATERIALS", "Total Material Cost"),
    ("CB_IS_PERSONNEL_EXPN", "Personnel Expenses"),
    ("IS_ADVERTISING_EXPENSES", "Advertising Expenses"),
    ("IS_OPEX_POWER_AND_FUEL", "Power & Fuel"),
    ("IS_TOT_OPER_EXP", "Total Operating Expenses"),
    ("IS_COMPARABLE_EBITDA", "EBITDA"),
    ("CF_DEPR_AMORT", "Depreciation & Amortization"),
    ("IS_COMPARABLE_EBIT", "Operating Income (EBIT)"),
    ("IS_FINANCE_COST", "Interest Expense"),
    ("IS_INT_INC", "Interest Income"),
    ("IS_COMP_PTP_EX_STK_BASED_COMP", "Pre-Tax Income"),
    ("IS_INC_TAX_EXP", "Income Tax Expense"),
    ("IS_COMP_NET_INCOME_GAAP", "Net Income (GAAP)"),
    ("IS_COMP_NET_INCOME_ADJUST_OLD", "Net Income (Adjusted)"),
    ("IS_COMP_EPS_ADJUSTED_OLD", "Adjusted EPS"),
    ("IS_COMP_GROSS_MARGIN_PERCENTAGE", "Gross Margin (%)"),
    ("EBITDA_TO_REVENUE", "EBITDA Margin (%)"),
    ("OPER_MARGIN", "Operating Margin (%)"),
    ("IS_PCT_SALES_CHANGE_VOLUME", "Volume Growth (%)"),
    ("IS_DIV_PER_SHR", "Dividends per Share"),
]
# lines appended to the annual Bloomberg income statement (missing from the standardized IS)
ANNUAL_EXTRA = ["Raw Materials & Consumables Used", "Purchases Of Merchandise",
                "Changes In Inventories Of Finished Goods and WiP", "Total Material Cost", "Advertising Expenses",
                "Power & Fuel", "Gross Margin (%)", "Volume Growth (%)"]
# expectation-vs-actual metrics
CONSENSUS_FIELDS = [
    ("IS_COMP_SALES", "Revenue"), ("IS_COMPARABLE_EBITDA", "EBITDA"), ("EBITDA_TO_REVENUE", "EBITDA Margin (%)"),
    ("IS_COMP_GROSS_MARGIN_PERCENTAGE", "Gross Margin (%)"), ("IS_COMPARABLE_EBIT", "Operating Income (EBIT)"),
    ("IS_COMP_PTP_EX_STK_BASED_COMP", "Pre-Tax Income"), ("IS_COMP_NET_INCOME_ADJUST_OLD", "Adjusted Net Income"),
    ("IS_COMP_EPS_ADJUSTED_OLD", "Adjusted EPS (₹)"),
    ("CB_IS_RAW_MATERIALS_AND_CONSUME_USED", "Raw Materials Consumed"), ("CB_IS_PERSONNEL_EXPN", "Personnel Expenses"),
    ("IS_ADVERTISING_EXPENSES", "Advertising Expenses"), ("IS_PCT_SALES_CHANGE_VOLUME", "Volume Growth (%)"),
    ("IS_DIV_PER_SHR", "Dividends per Share (₹)"), ("CB_CF_NET_CASH_OPERATING_ACT", "Cash Flow from Operations"),
    ("CB_CF_PURCHASES_OF_PPE", "Capital Expenditure"), ("CF_FREE_CASH_FLOW", "Free Cash Flow"),
]


def _period_from_header(text: str) -> tuple[pd.Timestamp, str] | None:
    m = re.search(r"Period:\s*(\d{4}):(Q[1-4]|Y)", str(text))
    if not m:
        return None
    fy, p = int(m.group(1)), m.group(2)
    if p == "Y":
        return pd.Timestamp(fy, 3, 31), "Y"
    q = int(p[1])
    year = fy if q == 4 else fy - 1
    return pd.Timestamp(year, QUARTER_END_MONTH[q], 1) + pd.offsets.MonthEnd(0), "Q"


def parse_consensus(sheets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Flatten all BEst sheets → one tidy frame:
    period, freq, code, metric, actual, mean, low, high, median, n_analysts"""
    rows, skipped = [], []
    for name, raw in sheets.items():
        raw = raw.reset_index(drop=True)
        if raw.empty or "Company Financial" not in str(raw.iloc[0, 0]):
            skipped.append(name)
            continue
        per = _period_from_header(raw.iloc[1, 0])
        if per is None:
            skipped.append(name)
            continue
        period, freq = per
        hdr = raw.iloc[2].tolist()
        idx = {str(v).strip(): j for j, v in enumerate(hdr) if pd.notna(v)}
        ca, cm = idx.get("Actuals", 4), idx.get("Mean Consensus", 5)
        cl, ch, cmed = idx.get("Low Consensus", 6), idx.get("High Consensus", 7), idx.get("Median Consensus", 8)
        broker_cols = list(range(max(ca, cm, cl, ch, cmed) + 1, raw.shape[1]))
        seen = set()
        for i in range(4, len(raw)):
            code = raw.iloc[i, 1]
            if not isinstance(code, str) or code in seen:
                continue
            seen.add(code)
            r = raw.iloc[i]
            num = lambda j: to_number(pd.Series([r.iloc[j]])).iloc[0] if j < len(r) else np.nan  # noqa: E731
            brokers = to_number(r.iloc[broker_cols]) if broker_cols else pd.Series(dtype=float)
            rows.append({"period": period, "freq": freq, "sheet": name, "code": code,
                         "metric": str(r.iloc[0]).strip(), "actual": num(ca), "mean": num(cm), "low": num(cl),
                         "high": num(ch), "median": num(cmed), "n_analysts": int(brokers.notna().sum())})
    df = pd.DataFrame(rows)
    if not df.empty:
        mon = df.apply(lambda x: is_monetary(x["metric"], x["code"]), axis=1)
        for c in ("actual", "mean", "low", "high", "median"):
            df.loc[mon, c] = df.loc[mon, c] * MILLION_TO_CRORE
        df["monetary"] = mon
    return df, {"sheets": len(sheets) - len(skipped), "skipped": skipped}


def consensus_datasets(cons: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build long-format datasets from the flattened consensus frame."""
    out: dict[str, pd.DataFrame] = {}
    if cons.empty:
        return out
    names = dict(CONSENSUS_FIELDS)
    for freq, key in (("Q", "brokerage_q"), ("Y", "brokerage_y")):
        d = cons[(cons["freq"] == freq) & cons["code"].isin(names)].copy()
        if freq == "Q":  # dividends are declared semi-annually/annually – quarterly comparisons are meaningless
            d = d[d["code"] != "IS_DIV_PER_SHR"]
        d["item"] = d["code"].map(names)
        order = {c: i for i, (c, _) in enumerate(CONSENSUS_FIELDS)}
        d = d.sort_values(["period", "code"], key=lambda s: s.map(order) if s.name == "code" else s)
        out[key] = d.rename(columns={"mean": "value", "actual": "value2"})[["period", "item", "value", "value2"]] \
            .dropna(subset=["value", "value2"], how="all").reset_index(drop=True)
        rng = d.rename(columns={"low": "value", "high": "value2"})[["period", "item", "value", "value2"]]
        cnt = d.assign(item=d["item"] + " #analysts", value=d["n_analysts"].astype(float), value2=np.nan)[
            ["period", "item", "value", "value2"]]
        out[f"consensus_range_{freq.lower()}"] = pd.concat([rng.dropna(subset=["value", "value2"], how="all"),
                                                            cnt[cnt["value"] > 0]], ignore_index=True)
    pnl = dict(PNL_FIELDS)
    q = cons[(cons["freq"] == "Q") & cons["code"].isin(pnl)].copy()
    q["item"] = q["code"].map(pnl)
    out["income_q"] = q.rename(columns={"actual": "value"}).assign(value2=np.nan)[
        ["period", "item", "value", "value2"]].dropna(subset=["value"]).reset_index(drop=True)
    y = cons[(cons["freq"] == "Y") & cons["code"].isin(pnl)].copy()
    y["item"] = y["code"].map(pnl)
    y = y[y["item"].isin(ANNUAL_EXTRA)]
    out["income_annual_extra"] = y.rename(columns={"actual": "value"}).assign(value2=np.nan)[
        ["period", "item", "value", "value2"]].dropna(subset=["value"]).reset_index(drop=True)
    return out


# ============================================================ one-shot loader
def load_workbooks(files: dict[str, bytes]) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """files = {filename: bytes}. Returns ({dataset_key: long_df}, report rows)."""
    datasets: dict[str, pd.DataFrame] = {}
    report: list[dict] = []
    estimates = []
    consensus_frames = []
    for fname, data in files.items():
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
        cons_sheets = {}
        for sname, raw in sheets.items():
            key, desc = classify_sheet(raw, sname)
            try:
                if key in ("usdinr", "brent", "brent_fut", "wti"):
                    df, meta = parse_prices(raw, "timeseries")
                elif key == "ohlc":
                    df, meta = parse_prices(raw, "ohlc")
                elif key in ("income", "balance", "cashflow"):
                    df, est, meta = parse_statement(raw)
                    if key == "income" and not est.empty:
                        estimates.append(est)
                elif key == "consensus":
                    cons_sheets[sname] = raw
                    continue
                else:
                    if desc != "empty sheet":
                        report.append({"File": fname, "Sheet": sname, "Dataset": "—", "Detail": desc, "Records": 0})
                    continue
            except Exception as e:  # noqa: BLE001
                report.append({"File": fname, "Sheet": sname, "Dataset": key, "Detail": f"ERROR: {e}", "Records": 0})
                continue
            datasets[key] = df
            span = f"{df['period'].min():%d-%b-%Y} → {df['period'].max():%d-%b-%Y}" if len(df) else "no rows"
            report.append({"File": fname, "Sheet": sname, "Dataset": key, "Detail": f"{desc}; {span}",
                           "Records": len(df)})
        if cons_sheets:
            cons, meta = parse_consensus(cons_sheets)
            consensus_frames.append(cons)
            report.append({"File": fname, "Sheet": f"{meta['sheets']} sheets", "Dataset": "consensus",
                           "Detail": "BEst consensus vs actuals: "
                                     f"{cons.loc[cons['freq'] == 'Q', 'period'].nunique() if len(cons) else 0} quarters, "
                                     f"{cons.loc[cons['freq'] == 'Y', 'period'].nunique() if len(cons) else 0} fiscal years",
                           "Records": len(cons)})
    if consensus_frames:
        cons = pd.concat(consensus_frames, ignore_index=True)
        built = consensus_datasets(cons)
        extra = built.pop("income_annual_extra", pd.DataFrame())
        datasets.update(built)
        if "income" in datasets and not extra.empty:
            inc = datasets["income"]
            extra = extra[~extra.set_index(["period", "item"]).index.isin(inc.set_index(["period", "item"]).index)]
            datasets["income"] = pd.concat([inc, extra], ignore_index=True)
            report.append({"File": "(merged)", "Sheet": "—", "Dataset": "income",
                           "Detail": f"Added {extra['item'].nunique()} cost lines (raw materials etc.) from consensus "
                                     f"actuals for {extra['period'].nunique()} fiscal years", "Records": len(extra)})
    if estimates:
        datasets["consensus_fy"] = pd.concat(estimates, ignore_index=True)
    return datasets, report


# ============================================================ save to database
UNITS = {"usdinr": "INR per USD", "brent": "USD per barrel", "brent_fut": "USD per barrel", "wti": "USD per barrel",
         "ohlc": "INR per share", "income": "₹ Crore (from INR mn)", "income_q": "₹ Crore (from INR mn)",
         "balance": "₹ Crore (from INR mn)", "cashflow": "₹ Crore (from INR mn)",
         "brokerage_q": "₹ Crore / ₹ per share / %", "brokerage_y": "₹ Crore / ₹ per share / %",
         "consensus_range_q": "₹ Crore / ₹ per share / %", "consensus_range_y": "₹ Crore / ₹ per share / %",
         "consensus_fy": "₹ Crore (from INR mn)"}


def save_all(datasets: dict[str, pd.DataFrame], filenames: str, make_current: bool | dict = True,
             notes: str = "Bloomberg import") -> list[dict]:
    """Persist every parsed dataset as a new version. Returns per-dataset results."""
    from . import database as db

    out = []
    for key, df in datasets.items():
        if df is None or df.empty:
            continue
        cur = make_current.get(key, False) if isinstance(make_current, dict) else make_current
        ds_id, v = db.save_dataset(key, df, "Bloomberg", filenames, UNITS.get(key, ""),
                                   {"importer": "bloomberg"}, notes, cur)
        out.append({"dataset": key, "version": v, "records": len(df), "current": bool(cur or v == 1)})
    return out


def autoload_if_empty() -> list[dict] | None:
    """If the database has no data at all and Bloomberg workbooks exist in data/input, load them."""
    from . import database as db
    from .config import INPUT_DIR

    if not db.list_versions().empty or not INPUT_DIR.exists():
        return None
    files = {p.name: p.read_bytes() for p in sorted(INPUT_DIR.glob("*.xls*")) if not p.name.startswith("~$")}
    if not files:
        return None
    datasets, _ = load_workbooks(files)
    res = save_all(datasets, ", ".join(files), True, "Auto-loaded from data/input on first run")
    db.log("autoload", f"{len(res)} datasets from {', '.join(files)}")
    return res
