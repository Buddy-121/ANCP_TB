"""Automated research report generation (HTML, Markdown, Excel data pack)."""
from __future__ import annotations

import html
import io
from datetime import datetime

import pandas as pd

from . import analytics as an
from . import data_access as da

CSS = """
body{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:1050px;margin:32px auto;color:#1b2430;line-height:1.5;padding:0 20px}
h1{color:#0b3d91;border-bottom:3px solid #0b3d91;padding-bottom:6px}
h2{color:#0b3d91;margin-top:34px;border-bottom:1px solid #d6dde8;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0}
th{background:#0b3d91;color:#fff;padding:6px 8px;text-align:left}
td{border-bottom:1px solid #e3e8ef;padding:5px 8px}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;color:#fff}
.FACT{background:#1f7a4d}.FORECAST{background:#b7791f}.SCENARIO{background:#6b46c1}.OPINION{background:#c53030}
.card{border:1px solid #d6dde8;border-radius:8px;padding:14px 18px;margin:14px 0}
.muted{color:#5a6475;font-size:12px}
@media print{h2{page-break-after:avoid}.card{page-break-inside:avoid}}
"""


def _fmt(v, pct=False, dec=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if isinstance(v, (int, float)):
        return f"{v:.{1 if pct else dec}%}" if pct else f"{v:,.{dec}f}"
    return html.escape(str(v))


def _table(rows: list[dict] | pd.DataFrame, pct_cols: set | None = None) -> str:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    if df.empty:
        return "<p class='muted'>No data.</p>"
    pct_cols = pct_cols or set()
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(
        f"<td>{_fmt(v, pct=(c in pct_cols or str(c).endswith('%')))}</td>" for c, v in r.items()) + "</tr>"
        for _, r in df.iterrows())
    return f"<table><tr>{head}</tr>{body}</table>"


def _tag(t: str) -> str:
    t = (t or "OPINION").split()[0].upper().strip(",")
    return f"<span class='tag {t}'>{t}</span>"


def build_html(ctx: dict, advisory: dict | None, sections: list[str]) -> str:
    p = [f"<html><head><meta charset='utf-8'><title>Asian Paints – Strategic Research Report</title>"
         f"<style>{CSS}</style></head><body>",
         f"<h1>Asian Paints Limited — Strategic Intelligence Report</h1>",
         f"<p class='muted'>Generated {ctx.get('generated_at')} · Evidence labels: {_tag('FACT')} calculated from data "
         f"{_tag('FORECAST')} statistical model output {_tag('SCENARIO')} assumption-based {_tag('OPINION')} AI / analyst judgement</p>"]

    if "Executive summary" in sections:
        p.append("<h2>1. Executive summary</h2>")
        if advisory and advisory.get("executive_summary"):
            p.append(f"<div class='card'>{_tag('OPINION')} {html.escape(advisory['executive_summary'])}</div>")
        st = (ctx.get("stock") or {}).get("statistics") or {}
        fin = ctx.get("financials") or {}
        hist = fin.get("history") or []
        last = hist[-1] if hist else {}
        rows = [{"Metric": "Share price (₹)", "Value": st.get("last_close"), "As of": st.get("last_date")},
                {"Metric": "1-year return", "Value": _fmt(st.get("return_1y"), True), "As of": st.get("last_date")},
                {"Metric": "1-year volatility (ann.)", "Value": _fmt(st.get("volatility_1y"), True), "As of": st.get("last_date")}]
        for k in ("Revenue", "EBITDA margin %", "Net margin %", "ROE %", "Revenue growth YoY %"):
            if k in last:
                rows.append({"Metric": k, "Value": _fmt(last[k], k.endswith("%")), "As of": last.get("index") or last.get("period")})
        p.append(f"{_tag('FACT')}" + _table(rows))

    if "Stock analysis" in sections and (ctx.get("stock") or {}).get("statistics"):
        p.append("<h2>2. Stock performance & technicals</h2>")
        st = ctx["stock"]["statistics"]
        p.append(_tag("FACT") + _table([{"Statistic": k, "Value": _fmt(v, pct=k.startswith(("return", "cagr", "annual", "volatility", "max", "current", "best", "worst")))}
                                        for k, v in st.items()]))
        if ctx["stock"].get("technicals"):
            p.append(_table([{"Indicator": k, "Value": _fmt(v)} for k, v in ctx["stock"]["technicals"].items()]))

    if "Macroeconomic analysis" in sections and ctx.get("macro"):
        m = ctx["macro"]
        p.append("<h2>3. Macroeconomic & commodity environment</h2>")
        rows = [{"Series": v["label"], "Latest": v["latest"], "Unit": v["unit"], "1M chg": _fmt(v["change_1m"], True),
                 "1Y chg": _fmt(v["change_1y"], True), "As of": v["as_of"]} for v in (m.get("series") or {}).values()]
        p.append(_tag("FACT") + _table(rows))
        if m.get("monthly_return_correlation_with_stock"):
            rows = [{"Driver": k, "Corr. with stock (monthly returns)": v["corr"], "p-value": v["p_value"],
                     "Best lead (months)": (m.get("lagged_macro_leads_stock") or {}).get(k, {}).get("best_lag_months")}
                    for k, v in m["monthly_return_correlation_with_stock"].items()]
            p.append(_table(rows) + f"<p class='muted'>{html.escape(m.get('method_note', ''))} n = {m.get('correlation_observations')}. Correlation ≠ causation.</p>")

    if "Financial analysis" in sections and (ctx.get("financials") or {}).get("history"):
        f = ctx["financials"]
        p.append("<h2>4. Financial statements & ratios</h2>")
        p.append(_tag("FACT") + _table(f["history"]))
        link = f.get("crude_to_margin_link") or {}
        if link and "error" not in link:
            p.append(f"<div class='card'>{_tag('FACT')} <b>Crude → gross margin link</b> ({link.get('frequency')}, n={link.get('n_periods')}): "
                     f"strongest association at lag {link.get('strongest_lag')} period(s), r = {_fmt(link.get('strongest_corr'))}, "
                     f"p = {_fmt(link.get('strongest_p'), dec=3)}. Estimated gross-margin change per +10% Brent (INR): "
                     f"{_fmt(link.get('gm_pp_per_10pct_crude'))} pp (slope p = {_fmt(link.get('slope_p'), dec=3)}).</div>")
        notes = f.get("notes") or {}
        if notes.get("assumptions"):
            p.append("<p class='muted'>Assumptions: " + " ".join(html.escape(a) for a in notes["assumptions"]) + "</p>")

    if "Brokerage expectations" in sections and ctx.get("brokerage"):
        p.append("<h2>5. Brokerage expectations vs actuals</h2>")
        for k, lbl in (("brokerage_q", "Quarterly"), ("brokerage_y", "Yearly")):
            if k in ctx["brokerage"]:
                p.append(f"<h3>{lbl}</h3>" + _tag("FACT") + _table(ctx["brokerage"][k]["summary"]))

    if "Scenarios & forecasts" in sections:
        p.append("<h2>6. Scenarios & forecasts</h2>")
        sc = ctx.get("scenarios") or {}
        if sc.get("scenarios"):
            rows = [{"Scenario": n, "Base EBITDA margin": _fmt(v["base_ebitda_margin"], True),
                     "New EBITDA margin": _fmt(v["new_ebitda_margin"], True), "Δ EBITDA (₹ Cr)": v["delta_ebitda"],
                     "≈Δ PAT (₹ Cr)": v["approx_delta_pat"], "Break-even price hike": _fmt(v["breakeven_price_increase"], True)}
                    for n, v in sc["scenarios"].items()]
            p.append(_tag("SCENARIO") + _table(rows) +
                     "<p class='muted'>Assumes 45% of material cost crude-linked, 10% other imports, volume elasticity −0.5 (editable in Predictive Analytics).</p>")
        for name, v in (ctx.get("forecasts") or {}).items():
            if isinstance(v, dict) and "forecast" in v:
                p.append(f"<h3>{html.escape(name)} – {html.escape(v['model'])}</h3>{_tag('FORECAST')}" + _table(v["forecast"]) +
                         f"<p class='muted'>Backtest MAPE {_fmt(v.get('backtest_mape'), True)}; beats naive: {v.get('beats_naive')}</p>")

    if "AI recommendations" in sections and advisory and advisory.get("recommendations"):
        p.append("<h2>7. Strategic recommendations</h2>")
        p.append(f"<p class='muted'>Generated by {html.escape(str((advisory.get('_meta') or {}).get('model')))} from the evidence pack above. "
                 "Recommendations are OPINION; supporting data is tagged by type.</p>")
        for i, r in enumerate(advisory["recommendations"], 1):
            sd = "".join(f"<li>{_tag(d.get('evidence_type'))} {html.escape(str(d.get('metric')))}: {html.escape(str(d.get('value')))} "
                         f"<span class='muted'>({html.escape(str(d.get('source')))})</span></li>" for d in r.get("supporting_data", []))
            p.append(f"<div class='card'><h3>{i}. {html.escape(r.get('area', ''))} — {html.escape(r.get('title', ''))}</h3>"
                     f"<p><b>Business problem:</b> {html.escape(r.get('business_problem', ''))}</p>"
                     f"<p><b>Supporting data:</b></p><ul>{sd}</ul>"
                     f"<p><b>Interpretation</b> {_tag('OPINION')}: {html.escape(r.get('analytical_interpretation', ''))}</p>"
                     f"<p><b>Recommended action</b> {_tag('OPINION')}: {html.escape(r.get('recommended_action', ''))}</p>"
                     f"<p><b>Expected impact:</b> {html.escape(r.get('expected_financial_impact', ''))}</p>"
                     f"<p><b>Timeframe:</b> {html.escape(r.get('implementation_timeframe', ''))} · "
                     f"<b>Confidence:</b> {html.escape(r.get('confidence', ''))} – {html.escape(r.get('confidence_rationale', ''))}</p>"
                     f"<p><b>Risks & limitations:</b> {html.escape(r.get('risks_and_limitations', ''))}</p></div>")

    p.append("<h2>Methodology & limitations</h2><ul>"
             "<li>Correlations and regressions use monthly log returns to avoid spurious level correlations; lagged tests use only past driver values.</li>"
             "<li>Forecasts are validated by expanding-window walk-forward backtests against a naive benchmark.</li>"
             "<li>Scenario outputs depend on stated assumptions (crude-linked share of materials, elasticity) and are not forecasts.</li>"
             "<li>Market data from Yahoo Finance is unofficial; Brent/WTI are front-month futures.</li>"
             "<li>Statistical association does not establish causation; small samples reduce reliability.</li></ul>")
    inv = ctx.get("data_inventory") or []
    p.append("<h2>Data sources</h2>" + _table(inv))
    p.append("</body></html>")
    return "\n".join(p)


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(v) for v in r.values) + " |")
    return "\n".join(out)


def build_markdown(ctx: dict, advisory: dict | None) -> str:
    lines = [f"# Asian Paints – Strategic Intelligence Report", f"_Generated {ctx.get('generated_at')}_", ""]
    if advisory and advisory.get("executive_summary"):
        lines += ["## Executive summary (AI OPINION)", advisory["executive_summary"], ""]
    st = (ctx.get("stock") or {}).get("statistics") or {}
    if st:
        lines += ["## Stock (FACT)"] + [f"- **{k}**: {v}" for k, v in st.items()] + [""]
    hist = (ctx.get("financials") or {}).get("history") or []
    if hist:
        lines += ["## Financial history (FACT)", _md_table(pd.DataFrame(hist)), ""]
    if advisory:
        for i, r in enumerate(advisory.get("recommendations", []), 1):
            lines += [f"## {i}. {r.get('area')} – {r.get('title')}",
                      f"- **Problem:** {r.get('business_problem')}",
                      "- **Supporting data:** " + "; ".join(f"[{d.get('evidence_type')}] {d.get('metric')}: {d.get('value')}" for d in r.get("supporting_data", [])),
                      f"- **Interpretation (OPINION):** {r.get('analytical_interpretation')}",
                      f"- **Action (OPINION):** {r.get('recommended_action')}",
                      f"- **Expected impact:** {r.get('expected_financial_impact')}",
                      f"- **Timeframe:** {r.get('implementation_timeframe')}",
                      f"- **Risks:** {r.get('risks_and_limitations')}",
                      f"- **Confidence:** {r.get('confidence')} – {r.get('confidence_rationale')}", ""]
    return "\n".join(lines)


def build_excel_pack() -> bytes:
    """All current datasets + computed ratios in one workbook."""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
        da.data_status().to_excel(xw, sheet_name="Inventory", index=False)
        ratios, _ = an.compute_ratios()
        if not ratios.empty:
            ratios.to_excel(xw, sheet_name="Computed ratios")
        for cat in ("income", "balance", "cashflow", "ratios"):
            w = da.statement_wide(cat)
            if not w.empty:
                w.columns = [c.strftime("%b-%Y") for c in w.columns]
                w.to_excel(xw, sheet_name=cat.title()[:31])
        for cat in ("usdinr", "brent", "wti"):
            s = da.price_series(cat)
            if not s.empty:
                s.to_frame("value").to_excel(xw, sheet_name=cat.upper())
        o = da.ohlc_frame()
        if not o.empty:
            o.to_excel(xw, sheet_name="OHLC")
        for cat in ("brokerage_q", "brokerage_y"):
            b = da.brokerage(cat)
            if not b.empty:
                an.brokerage_surprises(b).to_excel(xw, sheet_name=cat, index=False)
    return bio.getvalue()


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")
