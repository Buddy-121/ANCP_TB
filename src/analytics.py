"""Analytical functions. All functions are pure (inputs → outputs) so results can be
passed to the AI engine as structured, verifiable facts.

Statistical safeguards:
* Correlations/regressions use RETURNS (log changes), not price levels, by default —
  trending levels produce spurious correlations.
* Lagged analysis only uses PAST values of drivers (x shifted forward), never future.
* Regressions report HAC (Newey–West) standard errors, VIF and Durbin–Watson.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import data_access as da
from .config import TRADING_DAYS


# ============================================================ technical indicators
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    m = sma(s, n)
    sd = s.rolling(n, min_periods=n).std()
    return pd.DataFrame({"mid": m, "upper": m + k * sd, "lower": m - k * sd})


def log_returns(s: pd.Series) -> pd.Series:
    s = s[s > 0]
    return np.log(s).diff().dropna()


def hist_vol(s: pd.Series, window: int = 21) -> pd.Series:
    return log_returns(s).rolling(window).std() * np.sqrt(TRADING_DAYS)


def drawdown(s: pd.Series) -> pd.Series:
    return s / s.cummax() - 1


def price_stats(s: pd.Series, rf: float = 0.065) -> dict:
    """Summary statistics on a daily close series. rf = annual risk-free rate."""
    s = s.dropna()
    if len(s) < 30:
        return {}
    r = log_returns(s)
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)
    ann_ret = r.mean() * TRADING_DAYS
    last = s.index[-1]
    one_y = s[s.index >= last - pd.DateOffset(years=1)]

    def ret_since(off):
        base = s[s.index <= last - off]
        return s.iloc[-1] / base.iloc[-1] - 1 if len(base) else np.nan

    return {
        "last_close": float(s.iloc[-1]), "last_date": last.strftime("%Y-%m-%d"),
        "return_1m": ret_since(pd.DateOffset(months=1)), "return_3m": ret_since(pd.DateOffset(months=3)),
        "return_1y": ret_since(pd.DateOffset(years=1)), "return_3y": ret_since(pd.DateOffset(years=3)),
        "cagr_full_period": cagr, "annual_volatility": vol,
        "volatility_1y": float(log_returns(one_y).std() * np.sqrt(TRADING_DAYS)) if len(one_y) > 20 else np.nan,
        "sharpe": (ann_ret - rf) / vol if vol else np.nan,
        "max_drawdown": float(drawdown(s).min()), "current_drawdown": float(drawdown(s).iloc[-1]),
        "high_52w": float(one_y.max()), "low_52w": float(one_y.min()),
        "best_day": float(r.max()), "worst_day": float(r.min()),
        "skew": float(r.skew()), "excess_kurtosis": float(r.kurt()), "observations": int(len(s)),
        "years": round(years, 2),
    }


def technical_signals(s: pd.Series) -> dict:
    if len(s) < 200:
        return {}
    r = rsi(s).iloc[-1]
    m = macd(s).iloc[-1]
    s50, s200 = sma(s, 50).iloc[-1], sma(s, 200).iloc[-1]
    return {
        "rsi_14": float(r), "rsi_state": "Overbought (>70)" if r > 70 else "Oversold (<30)" if r < 30 else "Neutral",
        "macd": float(m["macd"]), "macd_signal": float(m["signal"]),
        "macd_state": "Bullish (MACD above signal)" if m["macd"] > m["signal"] else "Bearish (MACD below signal)",
        "sma_50": float(s50), "sma_200": float(s200),
        "trend": "Price above 200-DMA" if s.iloc[-1] > s200 else "Price below 200-DMA",
        "cross": "Golden cross regime (50>200)" if s50 > s200 else "Death cross regime (50<200)",
    }


# ============================================================ growth
def growth(s: pd.Series, months_back: int) -> pd.Series:
    """Growth vs the value exactly `months_back` months earlier (date-matched, gap safe)."""
    s = s.dropna().sort_index()
    idx = s.index + pd.offsets.MonthEnd(0)
    s = pd.Series(s.values, index=idx)
    prev_idx = (idx - pd.DateOffset(months=months_back)) + pd.offsets.MonthEnd(0)
    prev = s.reindex(prev_idx).values
    with np.errstate(divide="ignore", invalid="ignore"):
        g = pd.Series(s.values / prev - 1, index=idx)
    g[(prev <= 0) | pd.isna(prev)] = np.nan
    return g


def yoy(s: pd.Series) -> pd.Series:
    return growth(s, 12)


def qoq(s: pd.Series) -> pd.Series:
    return growth(s, 3)


def cagr(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return np.nan
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan


# ============================================================ alignment / correlation
def resample(s: pd.Series, freq: str = "ME", how: str = "last") -> pd.Series:
    r = s.resample(freq)
    return r.last() if how == "last" else r.mean()


def aligned_panel(series: dict[str, pd.Series], freq: str = "ME", how: str = "last") -> pd.DataFrame:
    df = pd.DataFrame({k: resample(v, freq, how) for k, v in series.items() if v is not None and len(v)})
    return df.dropna(how="all")


def to_changes(df: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    if method == "level":
        return df
    if method == "diff":
        return df.diff()
    return np.log(df.where(df > 0)).diff()


def corr_matrix(df: pd.DataFrame, method: str = "pearson") -> tuple[pd.DataFrame, pd.DataFrame, int]:
    d = df.dropna()
    c = d.corr(method=method)
    p = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for a in c.columns:
        for b in c.columns:
            if a != b and len(d) > 3:
                fn = sps.pearsonr if method == "pearson" else sps.spearmanr
                p.loc[a, b] = fn(d[a], d[b])[1]
    return c, p, len(d)


def lagged_corr(x: pd.Series, y: pd.Series, max_lag: int = 6) -> pd.DataFrame:
    """corr(x[t-k], y[t]) for k = 0..max_lag. Positive k = x LEADS y (uses only past x)."""
    rows = []
    for k in range(0, max_lag + 1):
        d = pd.concat([x.shift(k), y], axis=1).dropna()
        if len(d) > 5:
            r, p = sps.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
            rows.append({"lag": k, "corr": r, "p_value": p, "n": len(d), "significant_5pct": p < 0.05})
    return pd.DataFrame(rows)


def rolling_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    d = pd.concat([x, y], axis=1).dropna()
    return d.iloc[:, 0].rolling(window).corr(d.iloc[:, 1])


# ============================================================ regression
def ols_regression(y: pd.Series, X: pd.DataFrame, hac_lags: int = 3) -> dict:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(d) < X.shape[1] + 10:
        return {"error": f"Only {len(d)} overlapping observations – too few for a reliable regression."}
    Xc = sm.add_constant(d.drop(columns="y"))
    model = sm.OLS(d["y"], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    vif = {}
    if Xc.shape[1] > 2:
        for i, c in enumerate(Xc.columns):
            if c != "const":
                vif[c] = float(variance_inflation_factor(Xc.values, i))
    coef = pd.DataFrame({"coef": model.params, "std_err (HAC)": model.bse, "t": model.tvalues,
                         "p_value": model.pvalues, "ci_low": model.conf_int()[0], "ci_high": model.conf_int()[1]})
    jb_p = float(jarque_bera(model.resid)[1])
    return {
        "coefficients": coef, "r2": float(model.rsquared), "adj_r2": float(model.rsquared_adj),
        "n": int(model.nobs), "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "durbin_watson": float(durbin_watson(model.resid)), "jarque_bera_p": jb_p, "vif": vif,
        "fitted": model.fittedvalues, "resid": model.resid, "actual": d["y"],
    }


# ============================================================ fiscal aggregation
def fiscal_period_mean(daily: pd.Series, periods: pd.DatetimeIndex, freq: str) -> pd.Series:
    """Average of a daily series over each fiscal period ending at `periods`."""
    months = 12 if freq == "annual" else 3
    out = {}
    for p in periods:
        end = pd.Timestamp(p) + pd.offsets.MonthEnd(0)
        start = end - pd.DateOffset(months=months) + pd.Timedelta(days=1)
        w = daily[(daily.index >= start) & (daily.index <= end)]
        out[p] = w.mean() if len(w) > 10 else np.nan
    return pd.Series(out)


# ============================================================ financial ratios
def _ttm(s: pd.Series | None, quarterly: bool) -> pd.Series | None:
    if s is None or not quarterly:
        return s
    s = s.sort_index()
    return s.rolling(4, min_periods=4).sum()


def _avg(s: pd.Series | None) -> pd.Series | None:
    """Average of opening and closing balance (falls back to closing when no prior period)."""
    if s is None:
        return None
    return ((s + s.shift(1)) / 2).fillna(s)


def compute_ratios(income_cat: str = "income") -> tuple[pd.DataFrame, dict]:
    """Ratios derived from uploaded statements. Returns (ratios by period, notes).

    income_cat = "income" (annual) or "income_q" (quarterly P&L; BS ratios then use TTM flows)."""
    k = lambda cat, key: da.key_series(income_cat if cat == "income" else cat, key)  # noqa: E731
    rev = k("income", "revenue")
    notes: dict = {"missing_inputs": [], "assumptions": []}
    if rev is None:
        notes["missing_inputs"].append("Revenue (income statement)")
        return pd.DataFrame(), notes
    inc_cols = da.statement_wide(income_cat).columns
    quarterly = da.periodicity(inc_cols) == "quarterly"
    notes["income_periodicity"] = "quarterly" if quarterly else "annual"

    rm, pur, invc = k("income", "raw_material"), k("income", "purchases"), k("income", "inventory_change")
    emp, oth = k("income", "employee_cost"), k("income", "other_expenses")
    dep, fin = k("income", "depreciation"), k("income", "finance_cost")
    pbt, pat, oi = k("income", "pbt"), k("income", "net_profit"), k("income", "other_income")
    tax = k("income", "tax")
    ebitda_rep = k("income", "ebitda")

    r = pd.DataFrame(index=rev.index)
    r["Revenue"] = rev
    cogs = None
    if rm is not None:
        # COGS only where raw-material data exists (NaN elsewhere – never treated as zero)
        cogs = (rm + (pur.reindex(rm.index).fillna(0) if pur is not None else 0)
                + (invc.reindex(rm.index).fillna(0) if invc is not None else 0)).reindex(rev.index)
        r["Material cost (COGS)"] = cogs
        r["Gross margin %"] = (rev - cogs) / rev
        r["Material cost % revenue"] = cogs / rev
        if pur is None or invc is None:
            notes["assumptions"].append("COGS = materials consumed (+ purchases + inventory change where available).")
    else:
        notes["missing_inputs"].append("Raw material / COGS line")

    if all(x is not None for x in (cogs, emp, oth)) and cogs.notna().all():
        ebitda = rev - cogs - emp - oth
        notes["assumptions"].append("EBITDA = Revenue − COGS − Employee − Other expenses (excludes other income).")
    elif ebitda_rep is not None:
        ebitda = ebitda_rep
        notes["assumptions"].append("EBITDA = reported line item.")
    elif all(x is not None for x in (pbt, fin, dep)):
        ebitda = pbt + fin + dep - (oi if oi is not None else 0)
        notes["assumptions"].append("EBITDA = PBT + Finance cost + Depreciation − Other income.")
    else:
        ebitda = None
        notes["missing_inputs"].append("Inputs for EBITDA")
    if ebitda is not None:
        r["EBITDA"] = ebitda
        r["EBITDA margin %"] = ebitda / rev
    if pat is not None:
        r["Net profit"] = pat
        r["Net margin %"] = pat / rev
    if pbt is not None and tax is not None:
        r["Effective tax rate %"] = tax / pbt
    if pbt is not None and fin is not None:
        r["Interest coverage (x)"] = (pbt + fin) / fin.replace(0, np.nan)
    if emp is not None:
        r["Employee cost % revenue"] = emp / rev
    if oth is not None:
        r["Other expenses % revenue"] = oth / rev
    r["Revenue growth YoY %"] = yoy(rev).reindex(r.index + pd.offsets.MonthEnd(0)).values
    if pat is not None:
        r["PAT growth YoY %"] = yoy(pat).reindex(r.index + pd.offsets.MonthEnd(0)).values

    # ---- balance-sheet based (flows annualised by TTM when income is quarterly)
    bs = {key: k("balance", key) for key in ("total_assets", "total_equity", "total_debt", "current_assets",
                                              "current_liabilities", "inventory", "receivables", "payables", "cash",
                                              "current_investments")}
    if any(v is not None for v in bs.values()):
        if quarterly:
            notes["assumptions"].append("Income is quarterly: flows annualised as trailing-4-quarter sums for BS ratios.")
        rev_a, pat_a = _ttm(rev, quarterly), _ttm(pat, quarterly)
        cogs_a = _ttm(cogs, quarterly)
        ebit_a = _ttm(pbt + fin, quarterly) if (pbt is not None and fin is not None) else None
        b = pd.DataFrame({kk: v for kk, v in bs.items() if v is not None})
        j = b.join(pd.DataFrame({"rev": rev_a, "pat": pat_a, "cogs": cogs_a, "ebit": ebit_a}), how="inner")
        if not j.empty:
            if "total_equity" in j and j["pat"].notna().any():
                r.loc[j.index, "ROE %"] = j["pat"] / _avg(j["total_equity"])
            if "total_assets" in j:
                if j["pat"].notna().any():
                    r.loc[j.index, "ROA %"] = j["pat"] / _avg(j["total_assets"])
                r.loc[j.index, "Asset turnover (x)"] = j["rev"] / _avg(j["total_assets"])
            if "total_equity" in j and j["ebit"].notna().any():
                cap = j["total_equity"] + (j["total_debt"] if "total_debt" in j else 0)
                r.loc[j.index, "ROCE %"] = j["ebit"] / _avg(cap)
            if "total_equity" in j and "total_debt" in j:
                r.loc[j.index, "Debt / Equity (x)"] = j["total_debt"] / j["total_equity"]
            if "current_assets" in j and "current_liabilities" in j:
                r.loc[j.index, "Current ratio (x)"] = j["current_assets"] / j["current_liabilities"]
                if "inventory" in j:
                    r.loc[j.index, "Quick ratio (x)"] = (j["current_assets"] - j["inventory"]) / j["current_liabilities"]
            base_cost = j["cogs"] if j["cogs"].notna().any() else j["rev"]
            if "inventory" in j:
                r.loc[j.index, "Inventory days"] = j["inventory"] / base_cost * 365
            if "receivables" in j:
                r.loc[j.index, "Receivable days"] = j["receivables"] / j["rev"] * 365
            if "payables" in j:
                r.loc[j.index, "Payable days"] = j["payables"] / base_cost * 365
            if {"Inventory days", "Receivable days", "Payable days"} <= set(r.columns):
                r["Cash conversion cycle (days)"] = r["Inventory days"] + r["Receivable days"] - r["Payable days"]
            if j["cogs"].isna().all():
                notes["assumptions"].append("Inventory/payable days use revenue (COGS unavailable).")
            if "total_equity" in j and "total_assets" in j and j["pat"].notna().any():
                r.loc[j.index, "Equity multiplier (x)"] = _avg(j["total_assets"]) / _avg(j["total_equity"])
    else:
        notes["missing_inputs"].append("Balance sheet")

    # ---- cash flow based (annual statements only)
    cfo, capex, divs = k("cashflow", "cfo"), k("cashflow", "capex"), k("cashflow", "dividends")
    if cfo is not None and not quarterly:
        r = r.join(pd.DataFrame({"Cash from operations": cfo}), how="outer")
        if capex is not None:
            r.loc[capex.index.intersection(r.index), "Capex"] = capex.abs()
            r["Free cash flow"] = r["Cash from operations"] - r["Capex"]
            r["Capex % revenue"] = r["Capex"] / r["Revenue"]
            notes["assumptions"].append("Capex taken as absolute value of the capex line; FCF = CFO − Capex.")
        if pat is not None:
            r["CFO / PAT (x)"] = r["Cash from operations"] / r["Net profit"]
        if divs is not None and pat is not None:
            r.loc[divs.index.intersection(r.index), "Dividend payout %"] = divs.abs() / pat
    elif not quarterly:
        notes["missing_inputs"].append("Cash flow statement")
    r = r.sort_index()
    return r, notes


# ============================================================ crude → input cost link
def best_margin_source() -> str:
    """Quarterly P&L gives ~4x more observations for the crude→margin link; prefer it."""
    return "income_q" if da.has("income_q") and da.key_series("income_q", "raw_material") is not None else "income"


def margin_flags(income_cat: str, tol_pp: float = 2.0) -> pd.DataFrame:
    """Periods where the gross margin we compute (revenue − material cost) disagrees with the gross margin
    reported by the data vendor by more than `tol_pp` percentage points – usually a vendor data error
    (e.g. a revenue figure on a different consolidation basis)."""
    ratios, _ = compute_ratios(income_cat)
    w = da.statement_wide(income_cat)
    rep = next((i for i in w.index if i.lower().startswith("gross margin")), None)
    if ratios.empty or "Gross margin %" not in ratios or rep is None:
        return pd.DataFrame(columns=["period", "computed_pp", "reported_pp", "gap_pp"])
    d = pd.DataFrame({"computed_pp": ratios["Gross margin %"] * 100, "reported_pp": w.loc[rep]}).dropna()
    d["gap_pp"] = d["computed_pp"] - d["reported_pp"]
    comp = ratios["Gross margin %"].dropna() * 100
    neigh = comp.rolling(5, center=True, min_periods=3).median()
    d["outlier_vs_neighbours"] = (comp - neigh).abs().reindex(d.index) > 5
    return d[d["gap_pp"].abs() > tol_pp].rename_axis("period").reset_index()


def crude_cost_link(max_lag: int | None = None, income_cat: str | None = None,
                    exclude_flagged: bool = True) -> tuple[pd.DataFrame, dict]:
    """Relates Brent in INR (period average) to material cost % revenue and gross margin."""
    income_cat = income_cat or best_margin_source()
    ratios, _ = compute_ratios(income_cat)
    flags = margin_flags(income_cat)
    bad = flags.loc[flags["outlier_vs_neighbours"], "period"].tolist() if exclude_flagged and len(flags) else []
    if max_lag is None:
        max_lag = 3 if income_cat == "income_q" else 2
    brent, fx = da.price_series("brent"), da.price_series("usdinr")
    if ratios.empty or brent.empty or fx.empty or "Gross margin %" not in ratios:
        return pd.DataFrame(), {"error": "Needs income statement (revenue + material cost), Brent and USD/INR data."}
    freq = da.periodicity(ratios.index)
    brent_inr = (brent * fx.reindex(brent.index).ffill()).dropna()
    gm = ratios["Gross margin %"].copy()
    gm.loc[[p for p in bad if p in gm.index]] = np.nan  # keep the time grid; just blank the bad value
    df = pd.DataFrame({
        "Brent (USD) avg": fiscal_period_mean(brent, ratios.index, freq),
        "USD/INR avg": fiscal_period_mean(fx, ratios.index, freq),
        "Brent (INR) avg": fiscal_period_mean(brent_inr, ratios.index, freq),
        "Material cost % revenue": ratios["Material cost % revenue"].where(gm.notna()),
        "Gross margin %": gm,
    })
    df = df.dropna(subset=["Brent (INR) avg"])
    df = df.loc[df["Gross margin %"].first_valid_index():] if df["Gross margin %"].notna().any() else df.iloc[0:0]
    out = {"frequency": freq, "n_periods": int(df["Gross margin %"].notna().sum()), "source": income_cat,
           "excluded_periods": [p.strftime("%b-%Y") for p in bad]}
    if out["n_periods"] < 5:
        out["error"] = f"Only {out['n_periods']} overlapping periods – too few for inference."
        return df.dropna(), out
    # Quarterly: year-on-year changes remove seasonality; annual: year-on-year by construction.
    step = 4 if freq == "quarterly" else 1
    idx = pd.date_range(df.index.min(), df.index.max(), freq="QE-MAR" if freq == "quarterly" else "YE-MAR")
    g = df.reindex(idx)  # regular grid so shifts are true calendar lags
    chg = pd.DataFrame({"brent_inr_chg": np.log(g["Brent (INR) avg"]).diff(step),
                        "gm_chg_pp": g["Gross margin %"].diff(step) * 100})
    out["change_basis"] = "year-on-year change (4 quarters)" if step == 4 else "year-on-year change"
    out["lag_table"] = lagged_corr(chg["brent_inr_chg"], chg["gm_chg_pp"], max_lag)
    if not out["lag_table"].empty:
        best = out["lag_table"].iloc[out["lag_table"]["corr"].abs().argmax()]
        out["strongest_lag"] = int(best["lag"])
        out["strongest_corr"] = float(best["corr"])
        out["strongest_p"] = float(best["p_value"])
        X = pd.DataFrame({"brent_inr_chg": chg["brent_inr_chg"].shift(int(best["lag"]))})
        res = ols_regression(chg["gm_chg_pp"], X, hac_lags=step)
        if "error" not in res:
            out["gm_pp_per_10pct_crude"] = float(res["coefficients"].loc["brent_inr_chg", "coef"] * np.log(1.10))
            out["slope_p"] = float(res["coefficients"].loc["brent_inr_chg", "p_value"])
            out["slope_r2"] = float(res["r2"])
            out["slope_n"] = int(res["n"])
    df = df.dropna()
    out["level_corr_crude_vs_material_pct"] = float(df["Brent (INR) avg"].corr(df["Material cost % revenue"]))
    return df, out


# ============================================================ brokerage expectations
COST_METRIC_WORDS = ("raw material", "personnel", "advertising", "capital expenditure", "expense", "cost")


def is_cost_metric(name: str) -> bool:
    return any(w in str(name).lower() for w in COST_METRIC_WORDS)


def brokerage_surprises(df: pd.DataFrame, inline_band: float = 0.02) -> pd.DataFrame:
    """Surprise % = (Actual − Expected) / |Expected|. Within ±band = In-line.

    For cost lines (raw materials, personnel, advertising, capex) a LOWER actual than expected is
    favourable, so it is classified as a Beat."""
    d = df.dropna(subset=["expected", "actual"]).copy()
    d["surprise"] = d["actual"] - d["expected"]
    d["surprise_pct"] = d["surprise"] / d["expected"].abs().replace(0, np.nan)
    sign = np.where(d["metric"].map(is_cost_metric), -1, 1)
    fav = d["surprise_pct"] * sign
    d["result"] = np.where(fav > inline_band, "Beat", np.where(fav < -inline_band, "Miss", "In-line"))
    return d


def brokerage_summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in d.groupby("metric"):
        g = g.sort_values("period")
        streak, last = 0, g["result"].iloc[-1]
        for v in reversed(g["result"].tolist()):
            if v != last:
                break
            streak += 1
        rows.append({"Metric": m, "Periods": len(g), "Beats": int((g["result"] == "Beat").sum()),
                     "In-line": int((g["result"] == "In-line").sum()), "Misses": int((g["result"] == "Miss").sum()),
                     "Beat rate": float((g["result"] == "Beat").mean()),
                     "Avg surprise %": float(g["surprise_pct"].mean()),
                     "Median surprise %": float(g["surprise_pct"].median()),
                     "Latest period": g["period"].iloc[-1].strftime("%b-%Y"),
                     "Latest surprise %": float(g["surprise_pct"].iloc[-1]), "Latest result": last,
                     "Current streak": f"{streak} × {last}"})
    return pd.DataFrame(rows)
