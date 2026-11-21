"""Executive Dashboard – headline KPIs across market, fundamentals, macro and expectations."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import data_access as da
from src import ui
from src.config import DATASETS, MACRO_SERIES, SHORT_NAMES

ui.header("Executive Dashboard", "Asian Paints Limited · headline view of market performance, fundamentals, "
          "input-cost environment and street expectations")

status = da.data_status()
if not status["Loaded"].any():
    st.info("Welcome. No data is loaded yet. Start by uploading your datasets or fetching market data.")
    st.page_link(ui.DATA_PAGE, label="Open Data Management Centre", icon=":material/database:")
    st.stop()

st.markdown(ui.badge("FACT") + " All figures on this page are calculated from loaded data.")

# ------------------------------------------------------------------ market KPIs
stock = da.price_series("ohlc")
if not stock.empty:
    s = an.price_stats(stock)
    d1 = stock.iloc[-1] / stock.iloc[-2] - 1 if len(stock) > 1 else None
    st.markdown(f"##### Share price · as of {s.get('last_date', '')}")
    c = st.columns(5)
    with c[0]:
        ui.kpi("Last close", f"₹{s['last_close']:,.2f}", ui.pct(d1, 2, True) if d1 is not None else None)
    with c[1]:
        ui.kpi("1-year return", ui.pct(s["return_1y"]), help="Price return, excludes dividends")
    with c[2]:
        ui.kpi("52-week range", f"{s['low_52w']:,.0f}–{s['high_52w']:,.0f}", help="₹ per share")
    with c[3]:
        ui.kpi("Volatility (1Y, ann.)", ui.pct(s["volatility_1y"]), help="Std. dev. of daily log returns × √252")
    with c[4]:
        ui.kpi("From peak", ui.pct(s["current_drawdown"]), help=f"Max drawdown in data: {ui.pct(s['max_drawdown'])}")

# ------------------------------------------------------------------ fundamental KPIs
def _last(df: pd.DataFrame, col: str):
    """Latest valid value, its period and the change vs the previous valid value (in pp for ratios)."""
    if df.empty or col not in df:
        return None, None, None
    v = df[col].dropna()
    if v.empty:
        return None, None, None
    d = (v.iloc[-1] - v.iloc[-2]) if len(v) > 1 else None
    return v.iloc[-1], v.index[-1], d


def _kpi_row(df: pd.DataFrame, per_word: str, key: str):
    c = st.columns(5)
    rev, rp, _ = _last(df, "Revenue")
    g, _, _ = _last(df, "Revenue growth YoY %")
    with c[0]:
        ui.kpi("Revenue", ui.cr(rev), (ui.pct(g, 1, True) + " YoY") if g is not None else None,
               help=f"{per_word} ending {rp:%b-%Y}" if rp is not None else None)
    for i, (col, label) in enumerate([("Gross margin %", "Gross margin"), ("EBITDA margin %", "EBITDA margin"),
                                      ("Net margin %", "Net margin")], start=1):
        v, p_, d = _last(df, col)
        with c[i]:
            ui.kpi(label, ui.pct(v), f"{d * 100:+.1f} pp" if d is not None else None,
                   help=(f"{per_word} ending {p_:%b-%Y}; change vs previous {per_word.lower()}" if p_ is not None else None))
    with c[4]:
        if key == "annual":
            v, p_, d = _last(df, "ROE %")
            ui.kpi("ROE", ui.pct(v), f"{d * 100:+.1f} pp" if d is not None else None,
                   help=f"PAT / average equity, FY ending {p_:%b-%Y}" if p_ is not None else None)
        else:
            v, p_, d = _last(df, "Material cost % revenue")
            ui.kpi("Material cost % sales", ui.pct(v), f"{d * 100:+.1f} pp" if d is not None else None, inverse=True,
                   help="Raw materials + traded goods + inventory change, % of revenue")


ratios, notes = an.compute_ratios() if da.has("income") else (pd.DataFrame(), {})
if not ratios.empty:
    st.markdown(f"##### Fundamentals · fiscal year ending {ratios['Revenue'].dropna().index[-1]:%b-%Y}")
    _kpi_row(ratios, "Fiscal year", "annual")
    if ratios.get("Gross margin %") is not None and pd.isna(ratios["Gross margin %"].iloc[-1]):
        st.caption("Gross margin uses the latest year with raw-material data (hover ⓘ for the period).")
if da.has("income_q"):
    qr, _ = an.compute_ratios("income_q")
    if not qr.empty:
        st.markdown(f"##### Latest quarter · ending {qr['Revenue'].dropna().index[-1]:%b-%Y}")
        _kpi_row(qr, "Quarter", "quarterly")

# ------------------------------------------------------------------ macro KPIs
macro = [k for k in MACRO_SERIES if da.has(k)][:3]
brk = [k for k in ("brokerage_q", "brokerage_y") if da.has(k)]
if macro or brk:
    st.markdown("##### Input-cost environment & street expectations")
    c = st.columns(5)
    for i, k in enumerate(macro):
        ms = an.price_stats(da.price_series(k))
        unit = "₹" if k == "usdinr" else "$"
        with c[i]:
            ui.kpi(SHORT_NAMES[k], f"{unit}{ms['last_close']:,.2f}",
                   ui.pct(ms["return_1m"], 1, True) + " 1M" if pd.notna(ms.get("return_1m")) else None,
                   inverse=True, help=f"As of {ms['last_date']}. Rising = cost headwind (shown red).")
    for j, k in enumerate(brk[: 5 - len(macro)]):
        b = an.brokerage_surprises(da.brokerage(k))
        b = b[b["metric"].isin(["Revenue", "EBITDA", "Adjusted Net Income", "Adjusted EPS (₹)"])] \
            if b["metric"].isin(["Revenue", "EBITDA"]).any() else b
        if b.empty:
            continue
        lastp = b["period"].max()
        lb = b[b["period"] == lastp]
        beats = (lb["result"] == "Beat").sum()
        with c[len(macro) + j]:
            ui.kpi(("Qtr " if k == "brokerage_q" else "FY ") + f"{lastp:%b-%y} vs est.", f"{beats}/{len(lb)} beats",
                   f"{lb['surprise_pct'].mean():+.1%} avg",
                   help="Core metrics (revenue, EBITDA, adjusted net income, EPS) beating consensus by more than 2% in the latest " +
                        ("quarter" if k == "brokerage_q" else "year"))

st.divider()

# ------------------------------------------------------------------ charts
left, right = st.columns([3, 2])
with left:
    if not stock.empty:
        w = stock[stock.index >= stock.index[-1] - pd.DateOffset(years=3)]
        full = stock
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w.index, y=w.values, name="Close", line=dict(width=2, color=ui.color(0))))
        fig.add_trace(go.Scatter(x=w.index, y=an.sma(full, 50).reindex(w.index), name="50-DMA",
                                 line=dict(width=1.5, color=ui.color(1))))
        fig.add_trace(go.Scatter(x=w.index, y=an.sma(full, 200).reindex(w.index), name="200-DMA",
                                 line=dict(width=1.5, color=ui.color(2))))
        ui.chart(ui.style(fig, "Share price, last 3 years (₹)", 380), key="ex_px")
with right:
    if not ratios.empty and "EBITDA margin %" in ratios:
        rr = ratios.dropna(subset=["Revenue"]).tail(12)
        rr = rr.drop(columns=[c for c in ["Gross margin %"] if c in rr and rr[c].notna().sum() < 3])
        lbl = rr.index.strftime("%b-%y")
        fig = go.Figure(go.Bar(x=lbl, y=rr["Revenue"], marker_color=ui.color(0), name="Revenue",
                               hovertemplate="%{x}: ₹%{y:,.0f} Cr<extra></extra>"))
        ui.chart(ui.style(fig, "Revenue (₹ Cr)", 185, legend=False, hover="closest"), key="ex_rev")
        fig = go.Figure()
        for i, col in enumerate(["Gross margin %", "EBITDA margin %", "Net margin %"]):
            if col in rr:
                fig.add_trace(go.Scatter(x=lbl, y=rr[col], name=col.replace(" %", ""), mode="lines+markers",
                                         line=dict(width=2, color=ui.color(i)), marker=dict(size=8)))
        fig.update_yaxes(tickformat=".0%")
        ui.chart(ui.style(fig, "Margins", 195), key="ex_marg")

link_df, link = an.crude_cost_link()
if not link_df.empty and "error" not in link:
    st.markdown(f"##### Crude oil → gross margin ({link['frequency']})")
    c1, c2 = st.columns(2)
    lbl = link_df.index.strftime("%b-%y")
    with c1:
        fig = go.Figure(go.Scatter(x=lbl, y=link_df["Brent (INR) avg"], mode="lines+markers",
                                   line=dict(width=2, color=ui.color(1)), marker=dict(size=8), name="Brent ₹/bbl"))
        ui.chart(ui.style(fig, "Brent in ₹ per barrel (period average)", 280, legend=False), key="ex_bi")
    with c2:
        fig = go.Figure(go.Scatter(x=lbl, y=link_df["Gross margin %"], mode="lines+markers",
                                   line=dict(width=2, color=ui.color(0)), marker=dict(size=8), name="Gross margin"))
        fig.update_yaxes(tickformat=".1%")
        ui.chart(ui.style(fig, "Gross margin", 280, legend=False), key="ex_gm")
    if link.get("gm_pp_per_10pct_crude") is not None:
        sig = "statistically significant" if link.get("slope_p", 1) < 0.05 else "not statistically significant"
        st.caption(f"{ui.badge('FACT')} Across {link['n_periods']} {link['frequency']} periods, a 10% rise in Brent (₹) "
                   f"is associated with a {link['gm_pp_per_10pct_crude']:+.2f} pp change in gross margin at a "
                   f"{link['strongest_lag']}-period lag (p = {link['slope_p']:.3f}, {sig}). Association, not causation."
                   + (f" Excluded as data-quality outliers: {', '.join(link['excluded_periods'])}."
                      if link.get('excluded_periods') else ""))

with st.expander("Data freshness"):
    st.dataframe(status.drop(columns=["Key"]), hide_index=True, width="stretch")
