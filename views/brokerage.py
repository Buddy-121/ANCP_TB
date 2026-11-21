"""Brokerage Expectations vs Actuals – beat/miss analysis."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import data_access as da
from src import ui

ui.header("Brokerage Expectations vs Actuals", "Is Asian Paints beating or missing consensus – by how much, and how consistently?")
loaded = ui.require(["brokerage_q", "brokerage_y"], any_of=True)

c1, c2 = st.columns([1, 2])
cat = c1.radio("Frequency", loaded, horizontal=True, key="b_cat",
               format_func={"brokerage_q": "Quarterly", "brokerage_y": "Yearly"}.get)
band = c2.slider("In-line band (± %)", 0.0, 10.0, 2.0, 0.5, key="b_band",
                 help="Surprises within this band count as in-line") / 100

df = an.brokerage_surprises(da.brokerage(cat), band)
if df.empty:
    st.warning("No rows with both expected and actual values.")
    st.stop()
summary = an.brokerage_summary(df)

st.markdown(ui.badge("FACT") + " Surprise % = (Actual − Expected) / |Expected|. For cost lines (raw materials, "
            "personnel, advertising, capex) coming in **below** expectations counts as a beat.")
k = st.columns(4)
with k[0]:
    ui.kpi("Overall beat rate", ui.pct((df["result"] == "Beat").mean()), help="Share of metric-periods that beat")
with k[1]:
    ui.kpi("Miss rate", ui.pct((df["result"] == "Miss").mean()))
with k[2]:
    ui.kpi("Median surprise", ui.pct(df["surprise_pct"].median(), 1, True))
lp = df["period"].max()
with k[3]:
    l = df[df["period"] == lp]
    ui.kpi(f"Latest ({lp:%b-%Y})", f"{(l['result'] == 'Beat').sum()} beat · {(l['result'] == 'Miss').sum()} miss",
           ui.pct(l["surprise_pct"].mean(), 1, True) + " avg")

st.dataframe(summary.style.format({"Beat rate": "{:.0%}", "Avg surprise %": "{:+.1%}", "Median surprise %": "{:+.1%}",
                                   "Latest surprise %": "{:+.1%}"}), hide_index=True, width="stretch")

metrics = list(summary["Metric"])
metric = st.selectbox("Metric", metrics, key="b_m")
d = df[df["metric"] == metric].sort_values("period")
x = d["period"].dt.strftime("%b-%y")
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure([go.Bar(x=x, y=d["expected"], name="Expected", marker_color=ui.ink()["muted"]),
                     go.Bar(x=x, y=d["actual"], name="Actual", marker_color=ui.color(0))])
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    ui.chart(ui.style(fig, f"{metric}: expected vs actual", 360), key="b_ea")
with c2:
    col = np.where(d["result"] == "Beat", ui.GOOD, np.where(d["result"] == "Miss", ui.CRITICAL, ui.ink()["muted"]))
    fig = go.Figure(go.Bar(x=x, y=d["surprise_pct"], marker_color=col, customdata=d["result"],
                           hovertemplate="%{x}: %{y:+.1%} (%{customdata})<extra></extra>"))
    for y in (band, -band):
        fig.add_hline(y=y, line=dict(color=ui.ink()["axis"], dash="dot", width=1))
    fig.update_yaxes(tickformat=".0%")
    ui.chart(ui.style(fig, f"{metric}: surprise % (green beat · red miss · grey in-line)", 360, legend=False,
                      hover="closest"), key="b_sp")

rng_key = "consensus_range_q" if cat == "brokerage_q" else "consensus_range_y"
if da.has(rng_key):
    rg = da.brokerage(rng_key)
    band = rg[rg["metric"] == metric].rename(columns={"expected": "low", "actual": "high"})
    cnt = rg[rg["metric"] == f"{metric} #analysts"][["period", "expected"]].rename(columns={"expected": "analysts"})
    j = d.merge(band[["period", "low", "high"]], on="period", how="left").merge(cnt, on="period", how="left")
    if j["low"].notna().any():
        xx = j["period"].dt.strftime("%b-%y")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=xx, y=j["high"] - j["low"], base=j["low"], name="Analyst range (low–high)",
                             marker_color="rgba(137,135,129,0.35)", width=0.5,
                             hovertemplate="%{x}: %{base:,.1f} – %{y:,.1f} range<extra></extra>"))
        fig.add_trace(go.Scatter(x=xx, y=j["expected"], name="Mean consensus", mode="markers",
                                 marker=dict(symbol="line-ew", size=18, line=dict(width=2.5, color=ui.ink()["secondary"]))))
        fig.add_trace(go.Scatter(x=xx, y=j["actual"], name="Actual", mode="markers",
                                 marker=dict(size=9, color=np.where(j["result"] == "Beat", ui.GOOD,
                                                                    np.where(j["result"] == "Miss", ui.CRITICAL, ui.color(0))))))
        ui.chart(ui.style(fig, f"{metric}: where the actual landed within the analyst range", 360, hover="closest"),
                 key="b_rng")
        outside = ((j["actual"] > j["high"]) | (j["actual"] < j["low"])).sum()
        st.caption(f"{ui.badge('FACT')} Actual fell outside the full analyst range in **{outside} of {j['low'].notna().sum()}** "
                   f"periods. Median analysts covering: {j['analysts'].median():.0f}. A wide range = high uncertainty "
                   "about the period, so a 'miss' against the mean matters less.")
if any(k in metric for k in ("%",)):
    st.caption("For percentage metrics (margins, volume growth), surprise % is relative to the expected value; the "
               "absolute gap in percentage points is in the 'surprise' column.")

if len(metrics) > 1:
    df = df.assign(favourable=df["surprise_pct"] * np.where(df["metric"].map(an.is_cost_metric), -1, 1))
    hm = df.pivot_table(index="metric", columns="period", values="favourable")
    vals = np.abs(hm.values[~np.isnan(hm.values)])
    lim = float(np.clip(np.percentile(vals, 90), 0.03, 0.25)) if len(vals) else 0.1  # robust: outliers saturate
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns.strftime("%b-%y"), y=hm.index, zmin=-lim, zmax=lim,
                               colorscale=[[0, ui.CRITICAL], [0.5, ui.ink()["grid"]], [1, ui.GOOD]], xgap=2, ygap=2,
                               colorbar=dict(tickformat=".0%"),
                               hovertemplate="%{y} %{x}: %{z:+.1%} (favourable direction)<extra></extra>"))
    ui.chart(ui.style(fig, "Surprise heatmap – all metrics", 120 + 40 * len(hm), legend=False, hover="closest"),
             key="b_hm")

if da.has("ohlc") and cat == "brokerage_q":
    st.markdown("#### Share-price move in the month after each period end")
    px = da.price_series("ohlc")
    rows = []
    for p in sorted(df["period"].unique()):
        p = pd.Timestamp(p)
        a, b = px[px.index <= p + pd.Timedelta(days=30)], px[px.index <= p + pd.Timedelta(days=75)]
        if len(a) and len(b) and b.index[-1] > a.index[-1]:
            rows.append({"period": p, "price_move": b.iloc[-1] / a.iloc[-1] - 1})
    if rows:
        mv = pd.DataFrame(rows).merge(df[df["metric"] == metric][["period", "surprise_pct"]], on="period")
        if len(mv) > 4:
            r = mv["surprise_pct"].corr(mv["price_move"])
            st.caption(f"{ui.badge('FACT')} Correlation between {metric} surprise and the share move from 30 to 75 days "
                       f"after period end (≈ the results window): **{r:.2f}** (n = {len(mv)}). Approximation – exact result "
                       "dates are not in the dataset, so this proxy may include unrelated news.")

st.dataframe(d.assign(period=d["period"].dt.strftime("%b-%Y")).style.format(
    {"expected": "{:,.2f}", "actual": "{:,.2f}", "surprise": "{:+,.2f}", "surprise_pct": "{:+.1%}"}),
    hide_index=True, width="stretch")
