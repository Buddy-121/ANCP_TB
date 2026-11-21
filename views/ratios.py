"""Financial Ratio Analysis – computed from statements + uploaded ratio dataset."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import data_access as da
from src import ui

ui.header("Financial Ratio Analysis", "Profitability, returns, leverage, liquidity, working capital and cash quality")
loaded = ui.require(["income", "income_q", "ratios"], any_of=True)

GROUPS = {
    "Profitability": ["Gross margin %", "EBITDA margin %", "Net margin %", "Material cost % revenue",
                      "Employee cost % revenue", "Other expenses % revenue"],
    "Returns": ["ROE %", "ROCE %", "ROA %"],
    "Growth": ["Revenue growth YoY %", "PAT growth YoY %"],
    "Leverage & liquidity": ["Debt / Equity (x)", "Current ratio (x)", "Quick ratio (x)", "Interest coverage (x)"],
    "Working capital (days)": ["Inventory days", "Receivable days", "Payable days", "Cash conversion cycle (days)"],
    "Cash flow quality": ["CFO / PAT (x)", "Dividend payout %", "Capex % revenue"],
}

srcs = [k for k in ("income", "income_q") if da.has(k)]
INC = st.radio("Basis", srcs, horizontal=True, key="r_src",
               format_func={"income": "Annual statements", "income_q": "Quarterly P&L (BS ratios on TTM)"}.get) \
    if len(srcs) > 1 else (srcs[0] if srcs else "income")
tab_names = (["Computed ratios", "DuPont & working capital"] if srcs else []) + \
            (["Uploaded ratio dataset"] if da.has("ratios") else [])
tabs = dict(zip(tab_names, st.tabs(tab_names)))

if "Computed ratios" in tabs:
    ratios, notes = an.compute_ratios(INC)
    with tabs["Computed ratios"]:
        if ratios.empty:
            st.warning("Ratios could not be computed: " + ", ".join(notes["missing_inputs"]))
        else:
            st.markdown(ui.badge("FACT") + f" Computed from uploaded statements ({notes.get('income_periodicity')}).")
            n = st.slider("Periods shown", 3, max(3, len(ratios)), min(12, len(ratios)), key="r_n")
            rr = ratios.tail(n)
            x = rr.index.strftime("%b-%y")
            for g, cols in GROUPS.items():
                cols = [c for c in cols if c in rr and rr[c].notna().any()]
                if not cols:
                    continue
                fig = go.Figure()
                for i, c in enumerate(cols):
                    fig.add_trace(go.Scatter(x=x, y=rr[c], name=c.replace(" %", "").replace(" (x)", ""),
                                             mode="lines+markers", line=dict(width=2, color=ui.color(i)),
                                             marker=dict(size=8)))
                if any(c.endswith("%") for c in cols):
                    fig.update_yaxes(tickformat=".0%")
                ui.chart(ui.style(fig, g, 320), key=f"r_{g}")
            show = rr.T
            show.columns = [c.strftime("%b-%Y") for c in show.columns]
            st.dataframe(show.style.format(lambda v: "–" if pd.isna(v) else (f"{v:.1%}" if abs(v) < 5 else f"{v:,.1f}")),
                         width="stretch", height=520)
            st.caption("Values below 5 are shown as %, larger values as numbers (₹ Cr, days or ×) – refer to the row name.")
            if notes["missing_inputs"]:
                st.warning("Missing inputs: " + ", ".join(notes["missing_inputs"]) +
                           ". Check Data Management → Line-item mapping.")
            with st.expander("Definitions & assumptions"):
                st.markdown("\n".join(f"- {a}" for a in notes["assumptions"]) + """
- ROE = PAT / average equity; ROCE = (PBT + finance cost) / average (equity + debt); ROA = PAT / average assets.
- Inventory & payable days on COGS; receivable days on revenue; 365-day year.
- Average balances = mean of opening and closing; first period uses closing balance.
""")

    with tabs["DuPont & working capital"]:
        need = {"Net margin %", "Asset turnover (x)", "Equity multiplier (x)", "ROE %"}
        if not ratios.empty and need <= set(ratios.columns):
            d = ratios[list(need)].dropna().tail(10)
            x = d.index.strftime("%b-%y")
            st.markdown(ui.badge("FACT") + " **ROE = Net margin × Asset turnover × Equity multiplier**")
            c = st.columns(3)
            for i, (col, f) in enumerate([("Net margin %", ".1%"), ("Asset turnover (x)", ".2f"),
                                          ("Equity multiplier (x)", ".2f")]):
                with c[i]:
                    fig = go.Figure(go.Bar(x=x, y=d[col], marker_color=ui.color(i), name=col))
                    fig.update_yaxes(tickformat=f)
                    ui.chart(ui.style(fig, col, 280, legend=False, hover="closest"), key=f"dp_{i}")
            st.dataframe(d.T.rename(columns=lambda c: c.strftime("%b-%Y")).style.format("{:.3f}"), width="stretch")
        else:
            st.info("DuPont needs income statement and balance sheet (total assets & equity).")
        wc = [c for c in GROUPS["Working capital (days)"] if not ratios.empty and c in ratios]
        if len(wc) >= 3:
            d = ratios[wc].dropna(how="all").tail(10)
            x = d.index.strftime("%b-%y")
            fig = go.Figure()
            for i, c in enumerate(["Inventory days", "Receivable days"]):
                if c in d:
                    fig.add_trace(go.Bar(x=x, y=d[c], name=c, marker_color=ui.color(i)))
            if "Payable days" in d:
                fig.add_trace(go.Bar(x=x, y=-d["Payable days"], name="Payable days (financing)", marker_color=ui.color(2)))
            if "Cash conversion cycle (days)" in d:
                fig.add_trace(go.Scatter(x=x, y=d["Cash conversion cycle (days)"], name="Cash conversion cycle",
                                         mode="lines+markers", line=dict(width=2, color=ui.ink()["primary"]),
                                         marker=dict(size=8)))
            fig.update_layout(barmode="relative")
            ui.chart(ui.style(fig, "Working-capital cycle (days)", 380), key="wc")
            st.caption("CCC = inventory days + receivable days − payable days. Lower = less cash tied up in operations.")

if "Uploaded ratio dataset" in tabs:
    with tabs["Uploaded ratio dataset"]:
        w = da.statement_wide("ratios")
        st.markdown(ui.badge("FACT") + " Ratios as provided in your file (no unit conversion applied).")
        items = st.multiselect("Ratios to chart", list(w.index), default=list(w.index)[:3], key="ur_sel")
        if items:
            x = w.columns.strftime("%b-%y")
            fig = go.Figure([go.Scatter(x=x, y=w.loc[it], name=it, mode="lines+markers",
                                        line=dict(width=2, color=ui.color(i)), marker=dict(size=8))
                             for i, it in enumerate(items[:8])])
            ui.chart(ui.style(fig, "Selected ratios", 380), key="ur_ch")
            st.caption("Ratios with very different scales are clearer charted one at a time.")
        disp = w.copy()
        disp.columns = disp.columns.strftime("%b-%Y")
        st.dataframe(disp.style.format("{:,.2f}", na_rep="–"), width="stretch", height=480)
