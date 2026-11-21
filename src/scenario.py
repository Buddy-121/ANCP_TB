"""Scenario, sensitivity and DCF models. All assumptions are explicit parameters."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import analytics as an
from . import data_access as da


# ------------------------------------------------------------------ base case
def base_financials() -> dict:
    """Most recent 12-month financial base with revenue, material cost and EBITDA all available.

    Candidates: latest complete fiscal year (annual statements) and trailing-4-quarter sums
    (quarterly P&L). The more recent one is used."""
    cols = ["Revenue", "Material cost (COGS)", "EBITDA", "Net profit"]
    cands = []
    ann, notes = an.compute_ratios("income") if da.has("income") else (pd.DataFrame(), {})
    if not ann.empty and {"Revenue", "Material cost (COGS)", "EBITDA"} <= set(ann.columns):
        full = ann.dropna(subset=["Revenue", "Material cost (COGS)", "EBITDA"])
        if len(full):
            row = full[[c for c in cols if c in full]].iloc[-1].to_dict()
            row.update(basis=f"FY ending {full.index[-1]:%b-%Y}", end=full.index[-1], ratios=ann)
            cands.append(row)
    if da.has("income_q"):
        q, _ = an.compute_ratios("income_q")
        if not q.empty and {"Revenue", "Material cost (COGS)", "EBITDA"} <= set(q.columns):
            full = q.dropna(subset=["Revenue", "Material cost (COGS)", "EBITDA"]).tail(4)
            if len(full) == 4 and (full.index[-1] - full.index[0]).days < 300:
                row = full[[c for c in cols if c in full]].sum(min_count=4).to_dict()
                row.update(basis=f"TTM to {full.index[-1]:%b-%Y} (last 4 quarters)", end=full.index[-1], ratios=q)
                cands.append(row)
    if not cands:
        return {}
    base = max(cands, key=lambda r: r["end"])
    r = base.pop("ratios")
    last = r.dropna(subset=["Revenue"]).tail(12)
    base["tax_rate"] = float(np.clip(last["Effective tax rate %"].median(), 0.15, 0.35)) \
        if "Effective tax rate %" in last and last["Effective tax rate %"].notna().any() else 0.25
    base["capex_pct"] = float(ann["Capex % revenue"].dropna().tail(3).median()) \
        if not ann.empty and "Capex % revenue" in ann and ann["Capex % revenue"].notna().any() else None
    return base


# ------------------------------------------------------------------ cost scenario
@dataclass
class CostScenario:
    crude_change: float = 0.0          # % change in Brent (USD)
    fx_change: float = 0.0             # % change in USD/INR (+ = rupee depreciation)
    crude_linked_share: float = 0.45   # share of material cost linked to crude derivatives (ASSUMPTION)
    other_import_share: float = 0.10   # non-crude imported share of material cost (ASSUMPTION)
    pass_through_price: float = 0.0    # % price increase taken
    volume_elasticity: float = -0.5    # % volume change per 1% price increase (ASSUMPTION)
    other_cost_inflation: float = 0.0  # % change in employee + other expenses


def run_cost_scenario(base: dict, sc: CostScenario) -> dict:
    rev, cogs, ebitda = base["Revenue"], base["Material cost (COGS)"], base["EBITDA"]
    opex = rev - cogs - ebitda
    tax = base.get("tax_rate", 0.25)
    vol = 1 + sc.volume_elasticity * sc.pass_through_price
    price = 1 + sc.pass_through_price
    mat_index = (sc.crude_linked_share * (1 + sc.crude_change) * (1 + sc.fx_change)
                 + sc.other_import_share * (1 + sc.fx_change)
                 + (1 - sc.crude_linked_share - sc.other_import_share))
    new_rev = rev * price * vol
    new_cogs = cogs * mat_index * vol
    new_opex = opex * (1 + sc.other_cost_inflation)
    new_ebitda = new_rev - new_cogs - new_opex
    d_ebitda = new_ebitda - ebitda
    return {
        "base_revenue": rev, "new_revenue": new_rev, "base_cogs": cogs, "new_cogs": new_cogs,
        "base_ebitda": ebitda, "new_ebitda": new_ebitda, "delta_ebitda": d_ebitda,
        "base_ebitda_margin": ebitda / rev, "new_ebitda_margin": new_ebitda / new_rev,
        "base_gross_margin": (rev - cogs) / rev, "new_gross_margin": (new_rev - new_cogs) / new_rev,
        "approx_delta_pat": d_ebitda * (1 - tax), "material_cost_index": mat_index,
        "breakeven_price_increase": breakeven_price(base, sc), "assumptions": asdict(sc),
    }


def breakeven_price(base: dict, sc: CostScenario) -> float | None:
    """Price increase needed to restore the base EBITDA margin (bisection)."""
    target = base["EBITDA"] / base["Revenue"]
    lo, hi = -0.3, 0.5
    f = lambda p: run_margin(base, sc, p) - target  # noqa: E731
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def run_margin(base: dict, sc: CostScenario, price_change: float) -> float:
    s = CostScenario(**{**asdict(sc), "pass_through_price": price_change})
    rev, cogs, ebitda = base["Revenue"], base["Material cost (COGS)"], base["EBITDA"]
    opex = rev - cogs - ebitda
    vol = 1 + s.volume_elasticity * s.pass_through_price
    mat = (s.crude_linked_share * (1 + s.crude_change) * (1 + s.fx_change) + s.other_import_share * (1 + s.fx_change)
           + (1 - s.crude_linked_share - s.other_import_share))
    nr = rev * (1 + s.pass_through_price) * vol
    return (nr - cogs * mat * vol - opex * (1 + s.other_cost_inflation)) / nr


def sensitivity_grid(base: dict, sc: CostScenario, crude_range, fx_range) -> pd.DataFrame:
    rows = {}
    for f in fx_range:
        rows[f"{f:+.0%}"] = {f"{c:+.0%}": run_margin(base, CostScenario(**{**asdict(sc), "crude_change": c,
                                                                           "fx_change": f}), sc.pass_through_price)
                             for c in crude_range}
    g = pd.DataFrame(rows).T
    g.index.name = "USD/INR change"
    g.columns.name = "Brent change"
    return g


# ------------------------------------------------------------------ DCF
@dataclass
class DCFInputs:
    base_revenue: float
    growth_stage1: float = 0.10
    years_stage1: int = 5
    growth_stage2: float = 0.08
    years_stage2: int = 5
    ebitda_margin: float = 0.20
    dep_pct: float = 0.03
    capex_pct: float = 0.04
    nwc_pct_incr_rev: float = 0.10
    tax_rate: float = 0.25
    wacc: float = 0.11
    terminal_growth: float = 0.05
    net_cash: float = 0.0
    shares_cr: float = 95.92


def dcf(inp: DCFInputs) -> dict:
    if inp.wacc <= inp.terminal_growth:
        return {"error": "WACC must exceed terminal growth."}
    rows, rev = [], inp.base_revenue
    n = inp.years_stage1 + inp.years_stage2
    for t in range(1, n + 1):
        g = inp.growth_stage1 if t <= inp.years_stage1 else inp.growth_stage2
        new_rev = rev * (1 + g)
        ebitda = new_rev * inp.ebitda_margin
        dep = new_rev * inp.dep_pct
        ebit = ebitda - dep
        nopat = ebit * (1 - inp.tax_rate)
        capex = new_rev * inp.capex_pct
        dnwc = (new_rev - rev) * inp.nwc_pct_incr_rev
        fcff = nopat + dep - capex - dnwc
        df = 1 / (1 + inp.wacc) ** t
        rows.append({"Year": t, "Revenue": new_rev, "EBITDA": ebitda, "EBIT": ebit, "NOPAT": nopat,
                     "Capex": capex, "ΔNWC": dnwc, "FCFF": fcff, "Discount factor": df, "PV of FCFF": fcff * df})
        rev = new_rev
    tbl = pd.DataFrame(rows).set_index("Year")
    tv = tbl["FCFF"].iloc[-1] * (1 + inp.terminal_growth) / (inp.wacc - inp.terminal_growth)
    pv_tv = tv / (1 + inp.wacc) ** n
    ev = tbl["PV of FCFF"].sum() + pv_tv
    eq = ev + inp.net_cash
    return {"table": tbl, "terminal_value": tv, "pv_terminal": pv_tv, "enterprise_value": ev,
            "equity_value": eq, "per_share": eq / inp.shares_cr if inp.shares_cr else np.nan,
            "tv_share_of_ev": pv_tv / ev if ev else np.nan}


def dcf_sensitivity(inp: DCFInputs, waccs, gs) -> pd.DataFrame:
    out = {}
    for w in waccs:
        out[f"{w:.1%}"] = {f"{g:.1%}": (dcf(DCFInputs(**{**asdict(inp), "wacc": w, "terminal_growth": g}))
                                       .get("per_share", np.nan)) for g in gs}
    t = pd.DataFrame(out).T
    t.index.name = "WACC"
    t.columns.name = "Terminal growth"
    return t


def dcf_defaults() -> dict:
    ratios, _ = an.compute_ratios()
    d: dict = {}
    if ratios.empty:
        return d
    d["base_revenue"] = float(ratios["Revenue"].dropna().iloc[-1])
    d["base_basis"] = f"FY ending {ratios['Revenue'].dropna().index[-1]:%b-%Y}"
    if "EBITDA margin %" in ratios:
        d["ebitda_margin"] = float(ratios["EBITDA margin %"].dropna().tail(3).mean())
    if "Capex % revenue" in ratios and ratios["Capex % revenue"].notna().any():
        d["capex_pct"] = float(ratios["Capex % revenue"].dropna().tail(3).mean())
    dep = da.key_series("income", "depreciation")
    rev = da.key_series("income", "revenue")
    if dep is not None and rev is not None:
        d["dep_pct"] = float((dep / rev).dropna().tail(3).mean())
    tr = ratios["Effective tax rate %"].dropna().tail(3) if "Effective tax rate %" in ratios else pd.Series(dtype=float)
    d["tax_rate"] = float(np.clip(tr.median(), 0.15, 0.35)) if len(tr) else 0.25
    cash = da.key_series("balance", "cash")
    inv = da.key_series("balance", "current_investments")
    debt = da.key_series("balance", "total_debt")
    if cash is not None:
        nc = cash.dropna().iloc[-1]
        if inv is not None and inv.notna().any():
            nc += inv.dropna().iloc[-1]
        if debt is not None and debt.notna().any():
            nc -= debt.dropna().iloc[-1]
        d["net_cash"] = float(nc)
    sc = da.key_series("balance", "share_capital")
    if sc is not None and sc.notna().any():
        d["share_capital_cr"] = float(sc.dropna().iloc[-1])
    if "Revenue" in ratios:
        d["hist_rev_cagr"] = float(an.cagr(ratios["Revenue"].dropna().tail(6)))
    return {k: v for k, v in d.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
