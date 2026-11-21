"""Financial Statement Analysis – income statement, balance sheet, cash flow."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import data_access as da
from src import database as db
from src import ui
from src.config import DATASETS

ui.header("Financial Statement Analysis", "Trends, growth and common-size analysis of reported statements (₹ Crore)")
loaded = ui.require(["income", "income_q", "balance", "cashflow"], any_of=True)

c1, c2, c3 = st.columns([1.2, 1.2, 2])
cat = c1.selectbox("Statement", loaded, format_func=lambda k: DATASETS[k]["label"], key="f_cat")
w = da.statement_wide(cat)
freq = da.periodicity(w.columns)
views = ["Reported values", "YoY growth %"] + (["QoQ growth %"] if freq == "quarterly" else []) + ["Common-size %"]
view = c2.selectbox("View", views, key="f_view")
periods = list(w.columns)
labels = [p.strftime("%b-%Y") for p in periods]
rng = c3.select_slider("Periods", options=labels, value=(labels[max(0, len(labels) - 10)], labels[-1]), key="f_rng") \
    if len(labels) > 1 else (labels[0], labels[0])
sel = [p for p, l in zip(periods, labels) if labels.index(rng[0]) <= labels.index(l) <= labels.index(rng[1])]
meta = db.current_meta(cat)
st.caption(f"{ui.badge('FACT')} {freq.title()} data · source: {meta['filename'] or meta['source']} · "
           f"unit: {meta['unit'] or 'n/a'} → stored in ₹ Crore · version {meta['version']}")

resolved = da.resolved_items(cat)
base_key = {"income": "revenue", "income_q": "revenue", "balance": "total_assets", "cashflow": None}.get(cat)
if view == "Reported values":
    t = w[sel]
    fmt = lambda v: ui.indian(v, 2) if pd.notna(v) else "–"  # noqa: E731
elif view == "YoY growth %":
    t = w.apply(an.yoy, axis=1)
    t.columns = w.columns
    t = t[sel]
    fmt = lambda v: ui.pct(v)  # noqa: E731
elif view == "QoQ growth %":
    t = w.apply(an.qoq, axis=1)
    t.columns = w.columns
    t = t[sel]
    fmt = lambda v: ui.pct(v)  # noqa: E731
else:
    if base_key and resolved.get(base_key):
        t = w[sel].div(w.loc[resolved[base_key], sel])
        st.caption(f"Common-size base: **{resolved[base_key]}** = 100%")
    elif cat == "cashflow" and resolved.get("cfo"):
        t = w[sel].div(w.loc[resolved["cfo"], sel].abs())
        st.caption(f"Common-size base: **{resolved['cfo']}** (absolute) = 100%")
    else:
        st.warning("Base line item not identified – set it in Data Management → Line-item mapping.")
        t = w[sel] * float("nan")
    fmt = lambda v: ui.pct(v)  # noqa: E731

t_disp = t.copy()
t_disp.columns = [c.strftime("%b-%Y") for c in t_disp.columns]
search = st.text_input("Filter line items", key="f_q", placeholder="e.g. tax, inventories, borrowings")
if search:
    t_disp = t_disp[t_disp.index.str.contains(search, case=False, regex=False)]
st.dataframe(t_disp.style.format(fmt), width="stretch", height=430)
st.download_button("Download table (CSV)", t_disp.to_csv().encode(), f"{cat}_{view.split()[0].lower()}.csv",
                   "text/csv", key="f_dl")

st.markdown("#### Chart line items")
defaults = [v for v in resolved.values() if v][:3]
items = st.multiselect("Line items", list(w.index), default=defaults, key="f_items")
if items:
    sub = t.loc[items]
    x = [c.strftime("%b-%y") for c in sub.columns]
    fig = go.Figure()
    for i, it in enumerate(items[:8]):
        if view == "Reported values":
            fig.add_trace(go.Bar(x=x, y=sub.loc[it], name=it, marker_color=ui.color(i)))
        else:
            fig.add_trace(go.Scatter(x=x, y=sub.loc[it], name=it, mode="lines+markers",
                                     line=dict(width=2, color=ui.color(i)), marker=dict(size=8)))
    if view != "Reported values":
        fig.update_yaxes(tickformat=".0%")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    ui.chart(ui.style(fig, f"{view}" + (" (₹ Cr)" if view == "Reported values" else ""), 420), key="f_chart")
    if len(items) > 8:
        st.caption("Only the first 8 items are charted for readability.")

st.markdown("#### Growth summary of key lines")
rows = []
for key, item in resolved.items():
    if not item:
        continue
    s = w.loc[item].dropna()
    if len(s) < 2:
        continue
    g = an.yoy(s).dropna()
    rows.append({"Key metric": da.KEY_LABELS.get(key, key), "Line item": item, "Latest period": s.index[-1].strftime("%b-%Y"), "Latest value": s.iloc[-1],
                 "YoY": g.iloc[-1] if len(g) else None, "CAGR (full period)": an.cagr(s) if (s > 0).all() else None,
                 "Periods": len(s)})
if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"Latest value": lambda v: ui.indian(v, 0), "YoY": ui.pct, "CAGR (full period)": ui.pct},
                                 na_rep="–"), hide_index=True, width="stretch")
    st.caption("CAGR is shown only for series that are positive throughout. Mapping can be changed in Data Management.")
