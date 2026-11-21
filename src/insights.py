"""Builds the structured, Python-calculated evidence pack used by the Executive
Dashboard, the AI Advisory engine and the Research Report. Each section is tagged with
its evidence type so facts, forecasts and scenarios are never confused."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from . import analytics as an
from . import data_access as da
from . import forecasting as fc
from . import scenario as scn
from .config import COMPANY, DATASETS, MACRO_SERIES, SHORT_NAMES


def clean(o):
    """Make an object JSON-safe and compact (round floats, drop NaN)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        df = o.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime("%Y-%m-%d")
        return clean(df.reset_index().to_dict(orient="records"))
    if isinstance(o, pd.Series):
        s = o.copy()
        if isinstance(s.index, pd.DatetimeIndex):
            s.index = s.index.strftime("%Y-%m-%d")
        return clean(s.to_dict())
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 4)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 – record and continue, never crash the pack
        return {"error": f"{type(e).__name__}: {e}"}


def stock_section() -> dict:
    s = da.price_series("ohlc")
    if s.empty:
        return {"available": False}
    return {"evidence_type": "calculated_fact", "unit": "INR per share",
            "statistics": an.price_stats(s), "technicals": an.technical_signals(s)}


def macro_section() -> dict:
    out = {"evidence_type": "calculated_fact", "series": {}}
    for k in MACRO_SERIES:
        if not da.has(k):
            continue
        s = da.price_series(k)
        if s.empty:
            continue
        st = an.price_stats(s)
        out["series"][k] = {"label": DATASETS[k]["label"], "unit": DATASETS[k]["unit"],
                            "latest": st.get("last_close"), "as_of": st.get("last_date"),
                            "change_1m": st.get("return_1m"), "change_1y": st.get("return_1y"),
                            "volatility_1y": st.get("volatility_1y"), "high_52w": st.get("high_52w"),
                            "low_52w": st.get("low_52w")}
    b, fx = da.price_series("brent"), da.price_series("usdinr")
    if not b.empty and not fx.empty:
        bi = (b * fx.reindex(b.index).ffill()).dropna()
        out["brent_in_inr"] = {"latest": bi.iloc[-1], "change_1y": bi.iloc[-1] / bi[bi.index <= bi.index[-1] - pd.DateOffset(years=1)].iloc[-1] - 1
                               if (bi.index <= bi.index[-1] - pd.DateOffset(years=1)).any() else None,
                               "unit": "INR per barrel"}
    stock = da.price_series("ohlc")
    panel = an.aligned_panel({"stock": stock, **{k: da.price_series(k) for k in MACRO_SERIES if da.has(k)}})
    if "stock" in panel and panel.shape[1] > 1:
        rets = an.to_changes(panel, "log")
        c, p, n = an.corr_matrix(rets)
        out["monthly_return_correlation_with_stock"] = {
            k: {"corr": c.loc["stock", k], "p_value": p.loc["stock", k]} for k in c.columns if k != "stock"}
        out["correlation_observations"] = n
        lags = {}
        for k in c.columns:
            if k == "stock":
                continue
            lt = an.lagged_corr(rets[k], rets["stock"], 3)
            if not lt.empty:
                best = lt.iloc[lt["corr"].abs().argmax()]
                lags[k] = {"best_lag_months": int(best["lag"]), "corr": best["corr"], "p_value": best["p_value"]}
        out["lagged_macro_leads_stock"] = lags
        out["method_note"] = "Monthly log returns (month-end); Pearson; lag k = macro k months before stock."
    return out


def financial_section() -> dict:
    ratios, notes = an.compute_ratios()
    if ratios.empty:
        return {"available": False, "notes": notes}
    keep = [c for c in ["Revenue", "EBITDA", "Net profit", "Gross margin %", "Material cost % revenue",
                        "EBITDA margin %", "Net margin %", "Revenue growth YoY %", "PAT growth YoY %", "ROE %",
                        "ROCE %", "Debt / Equity (x)", "Current ratio (x)", "Inventory days", "Receivable days",
                        "Payable days", "Cash conversion cycle (days)", "Free cash flow", "CFO / PAT (x)",
                        "Dividend payout %", "Capex % revenue"] if c in ratios]
    hist = ratios[keep].dropna(how="all").tail(6)
    out = {"evidence_type": "calculated_fact", "unit": "₹ Crore for amounts; ratios as decimals",
           "periodicity": notes.get("income_periodicity"), "history": hist, "notes": notes}
    if "Revenue" in ratios:
        out["revenue_cagr_last_5_periods"] = an.cagr(ratios["Revenue"].dropna().tail(6))
    link_df, link = an.crude_cost_link()
    link.pop("lag_table", None)
    if da.has("income_q"):
        q, _ = an.compute_ratios("income_q")
        qk = [c for c in ["Revenue", "Gross margin %", "Material cost % revenue", "EBITDA margin %", "Net margin %",
                          "Revenue growth YoY %"] if c in q]
        out["quarterly_recent"] = {"evidence_type": "calculated_fact", "source": "Bloomberg quarterly actuals",
                                   "history": q[qk].dropna(how="all").tail(8)}
    out["crude_to_margin_link"] = {"evidence_type": "calculated_fact (statistical association, not causation)",
                                   **link}
    return out


def brokerage_section() -> dict:
    out = {"evidence_type": "calculated_fact", "definition": "surprise % = (actual − expected)/|expected|; ±2% = in-line"}
    for k in ("brokerage_q", "brokerage_y"):
        df = da.brokerage(k)
        if df.empty:
            continue
        s = an.brokerage_surprises(df)
        out[k] = {"summary": an.brokerage_summary(s),
                  "latest_period": s.sort_values("period").groupby("metric").tail(1)[
                      ["metric", "period", "expected", "actual", "surprise_pct", "result"]]}
    return out


def scenario_section() -> dict:
    base = scn.base_financials()
    if not base or any(pd.isna(base.get(k)) for k in ("Revenue", "Material cost (COGS)", "EBITDA")):
        return {"available": False}
    out = {"evidence_type": "assumption-based scenario (model output, not a forecast)", "base": base,
           "scenarios": {}}
    for name, sc in {
        "Brent +20%, INR −5%": scn.CostScenario(crude_change=0.20, fx_change=0.05),
        "Brent −15%": scn.CostScenario(crude_change=-0.15),
        "Brent +20%, INR −5%, 3% price hike": scn.CostScenario(crude_change=0.20, fx_change=0.05,
                                                               pass_through_price=0.03),
    }.items():
        r = scn.run_cost_scenario(base, sc)
        out["scenarios"][name] = {k: r[k] for k in ("new_ebitda_margin", "base_ebitda_margin", "delta_ebitda",
                                                    "approx_delta_pat", "new_gross_margin",
                                                    "breakeven_price_increase", "assumptions")}
    return out


def forecast_section() -> dict:
    out = {"evidence_type": "forecast (statistical, with uncertainty)"}
    for k, label in (("brent", "Brent"), ("usdinr", "USD/INR")):
        s = da.price_series(k)
        if len(s) < 300:
            continue
        m = an.resample(s, "ME", "last").dropna()
        bt = fc.walk_forward(m, "ETS (damped trend)", 12, log=True)
        f = fc.forecast(m, 6, "ETS (damped trend)", "ME", log=True)
        out[label] = {"model": "ETS damped trend on log monthly close", "horizon_months": 6,
                      "forecast": f.round(3), "backtest_mape": bt.get("metrics", {}).get("MAPE"),
                      "beats_naive": bt.get("beats_naive")}
    return out


def build_context(include_forecasts: bool = True) -> dict:
    ctx = {
        "company": COMPANY, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "currency_note": "Financial statement amounts are in ₹ Crore after unit normalisation.",
        "data_inventory": da.data_status()[["Dataset", "Loaded", "From", "To", "Source", "Updated"]],
        "stock": _safe(stock_section), "macro": _safe(macro_section),
        "financials": _safe(financial_section), "brokerage": _safe(brokerage_section),
        "scenarios": _safe(scenario_section),
    }
    if include_forecasts:
        ctx["forecasts"] = _safe(forecast_section)
    return clean(ctx)
