"""Asian Paints Stock Analysis – technicals, risk and competitor benchmarking."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import analytics as an
from src import data_access as da
from src import market_data as md
from src import ui
from src.config import BENCHMARK_INDEX, PEERS, TRADING_DAYS

ui.header("Asian Paints Stock Analysis", "Price action, technical indicators, risk statistics and peer benchmarking")
ui.require(["ohlc"])

o = da.ohlc_frame()
close = o["close"].dropna()
start, end = ui.date_range(close.index, "stock", "3Y")
ow = o.loc[start:end]
cw = close.loc[start:end]

stt = an.price_stats(cw)
tech = an.technical_signals(close.loc[:end])
k = st.columns(6)
with k[0]:
    ui.kpi("Close", f"₹{cw.iloc[-1]:,.2f}", ui.pct(cw.iloc[-1] / cw.iloc[-2] - 1, 2, True) if len(cw) > 1 else None)
with k[1]:
    ui.kpi("Period return", ui.pct(cw.iloc[-1] / cw.iloc[0] - 1))
with k[2]:
    ui.kpi("Ann. volatility", ui.pct(stt.get("annual_volatility")))
with k[3]:
    ui.kpi("Max drawdown", ui.pct(stt.get("max_drawdown")))
with k[4]:
    ui.kpi("RSI (14)", ui.num(tech.get("rsi_14"), 1), tech.get("rsi_state"), help="Wilder's RSI", neutral=True)
with k[5]:
    ui.kpi("Sharpe ratio", ui.num(stt.get("sharpe")), help="Risk-free rate 6.5%. " + "(Ann. log return − 6.5%) / ann. volatility")

t1, t2, t3 = st.tabs(["Price & technicals", "Returns & risk", "Competitor benchmarking"])

with t1:
    c1, c2, c3, c4 = st.columns(4)
    ctype = c1.selectbox("Chart", ["Candlestick", "Line"], key="s_ct")
    smas = c2.multiselect("Moving averages", [20, 50, 100, 200], default=[50, 200], key="s_sma")
    boll = c3.toggle("Bollinger bands (20, 2σ)", key="s_bb")
    show_vol = c4.toggle("Volume", value="volume" in o, key="s_vol", disabled="volume" not in o)
    rows = 3 + (1 if show_vol else 0)
    heights = [0.5, 0.15, 0.17, 0.18] if show_vol else [0.6, 0.2, 0.2]
    titles = [""] + (["Volume"] if show_vol else []) + ["RSI (14)", "MACD (12, 26, 9)"]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=heights,
                        subplot_titles=titles)
    if ctype == "Candlestick" and {"open", "high", "low"} <= set(ow.columns):
        fig.add_trace(go.Candlestick(x=ow.index, open=ow["open"], high=ow["high"], low=ow["low"], close=ow["close"],
                                     name="OHLC", increasing_line_color=ui.GOOD, decreasing_line_color=ui.CRITICAL),
                      row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=cw.index, y=cw, name="Close", line=dict(width=2, color=ui.color(0))), row=1, col=1)
    for i, n in enumerate(smas):
        fig.add_trace(go.Scatter(x=cw.index, y=an.sma(close, n).loc[start:end], name=f"{n}-DMA",
                                 line=dict(width=1.4, color=ui.color(i + 1))), row=1, col=1)
    if boll:
        b = an.bollinger(close).loc[start:end]
        fig.add_trace(go.Scatter(x=b.index, y=b["upper"], name="BB upper", line=dict(width=1, color=ui.ink()["muted"])), 1, 1)
        fig.add_trace(go.Scatter(x=b.index, y=b["lower"], name="BB lower", fill="tonexty",
                                 fillcolor="rgba(137,135,129,0.12)", line=dict(width=1, color=ui.ink()["muted"])), 1, 1)
    r = 2
    if show_vol:
        fig.add_trace(go.Bar(x=ow.index, y=ow["volume"], name="Volume", marker_color=ui.ink()["muted"]), r, 1)
        r += 1
    rs = an.rsi(close).loc[start:end]
    fig.add_trace(go.Scatter(x=rs.index, y=rs, name="RSI", line=dict(width=1.5, color=ui.color(6))), r, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(color=ui.ink()["axis"], dash="dot", width=1), row=r, col=1)
    m = an.macd(close).loc[start:end]
    fig.add_trace(go.Bar(x=m.index, y=m["hist"], name="MACD hist",
                         marker_color=np.where(m["hist"] >= 0, ui.GOOD, ui.CRITICAL)), r + 1, 1)
    fig.add_trace(go.Scatter(x=m.index, y=m["macd"], name="MACD", line=dict(width=1.5, color=ui.color(0))), r + 1, 1)
    fig.add_trace(go.Scatter(x=m.index, y=m["signal"], name="Signal", line=dict(width=1.5, color=ui.color(1))), r + 1, 1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig = ui.style(fig, None, 820)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_annotations(font_size=12)
    ui.chart(fig, key="s_main")
    if tech:
        st.markdown(f"{ui.badge('FACT')} **{tech['trend']}** · **{tech['cross']}** · MACD: {tech['macd_state']} · "
                    f"RSI {tech['rsi_14']:.1f} ({tech['rsi_state']})")
    with st.expander("Indicator definitions & limitations"):
        st.markdown("""
- **SMA/DMA** – simple moving average of closes. **Bollinger** – 20-day SMA ± 2 standard deviations.
- **RSI (14)** – Wilder smoothing of average gains/losses; >70 overbought, <30 oversold.
- **MACD** – EMA(12) − EMA(26); signal = EMA(9) of MACD.
- Technical indicators describe past price behaviour. They are not evidence of fundamental value and have weak,
  unstable predictive power for large-cap stocks. Prices are unadjusted unless your file contains adjusted prices.
""")

with t2:
    rets = an.log_returns(cw)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(x=rets, nbinsx=80, marker_color=ui.color(0), name="Daily returns"))
        fig.update_xaxes(tickformat=".1%")
        ui.chart(ui.style(fig, "Distribution of daily log returns", 330, legend=False, hover="closest"), key="s_hist")
        st.caption(f"Skew {stt.get('skew', np.nan):.2f} · excess kurtosis {stt.get('excess_kurtosis', np.nan):.2f} – "
                   "fat tails mean normal-distribution risk measures understate extreme moves.")
    with c2:
        dd = an.drawdown(cw)
        fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(width=1.5, color=ui.CRITICAL), name="Drawdown"))
        fig.update_yaxes(tickformat=".0%")
        ui.chart(ui.style(fig, "Drawdown from running peak", 330, legend=False), key="s_dd")
    hv = pd.DataFrame({"21-day": an.hist_vol(close, 21), "63-day": an.hist_vol(close, 63),
                       "252-day": an.hist_vol(close, 252)}).loc[start:end]
    fig = go.Figure([go.Scatter(x=hv.index, y=hv[c], name=c, line=dict(width=1.6, color=ui.color(i)))
                     for i, c in enumerate(hv.columns)])
    fig.update_yaxes(tickformat=".0%")
    ui.chart(ui.style(fig, "Historical volatility (annualised)", 320), key="s_hv")
    yr = close.resample("YE").last().pct_change().dropna()
    yr = yr[yr.index >= start - pd.DateOffset(years=1)]
    fig = go.Figure(go.Bar(x=yr.index.year, y=yr, marker_color=np.where(yr >= 0, ui.GOOD, ui.CRITICAL),
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    ui.chart(ui.style(fig, "Calendar-year price return", 300, legend=False, hover="closest"), key="s_yr")
    PCT = ("return", "cagr", "annual", "volatility", "max", "current", "best", "worst")
    st.dataframe(pd.DataFrame([{"Statistic": k.replace("_", " ").capitalize(),
                                "Value": ui.pct(v) if k.startswith(PCT) else (ui.num(v) if isinstance(v, float) else str(v))}
                               for k, v in stt.items()]), hide_index=True, width="stretch")

with t3:
    st.caption("Peer and index prices are fetched live from Yahoo Finance (not stored). Fundamentals snapshot uses "
               "Yahoo `info`, which can be incomplete for Indian stocks – verify against annual reports.")

    @st.cache_data(ttl=3600, show_spinner="Fetching peer prices…")
    def peers_panel(period):
        return md.fetch_close_panel({**PEERS, "Nifty 50": BENCHMARK_INDEX}, period)

    @st.cache_data(ttl=3600, show_spinner="Fetching fundamentals…")
    def peer_fund():
        return md.fetch_fundamentals(PEERS)

    c1, c2 = st.columns([1, 3])
    period = c1.selectbox("Lookback", ["1y", "3y", "5y", "10y"], index=2, key="p_per")
    if c2.button("Load peer comparison", type="primary", key="p_go"):
        st.session_state["peer_period"] = period
    if "peer_period" in st.session_state:
        try:
            panel = peers_panel(st.session_state["peer_period"])
        except Exception as e:  # noqa: BLE001
            panel = pd.DataFrame()
            st.error(f"Fetch failed: {e}")
        if panel.attrs.get("errors"):
            st.warning("Could not fetch: " + ", ".join(panel.attrs["errors"]))
        if not panel.empty:
            panel = panel.ffill().dropna(how="all")
            idx = panel / panel.bfill().iloc[0] * 100
            fig = go.Figure([go.Scatter(x=idx.index, y=idx[c], name=c,
                                        line=dict(width=2.4 if c == "Asian Paints" else 1.4, color=ui.color(i)))
                             for i, c in enumerate(idx.columns)])
            ui.chart(ui.style(fig, "Total price performance, indexed to 100", 420), key="p_idx")
            rows = []
            r = np.log(panel).diff()
            mkt = r.get("Nifty 50")
            for c in panel.columns:
                s = panel[c].dropna()
                rr = r[c].dropna()
                beta = np.nan
                if mkt is not None and c != "Nifty 50":
                    d = pd.concat([rr, mkt], axis=1).dropna()
                    beta = np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var() if len(d) > 30 else np.nan
                rows.append({"Company": c, "Return": s.iloc[-1] / s.iloc[0] - 1, "CAGR": an.cagr(s),
                             "Ann. volatility": rr.std() * np.sqrt(TRADING_DAYS),
                             "Max drawdown": an.drawdown(s).min(), "Beta vs Nifty": beta,
                             "Corr. with Asian Paints": rr.corr(r["Asian Paints"]) if "Asian Paints" in r else np.nan})
            tbl = pd.DataFrame(rows)
            st.dataframe(tbl.style.format({"Return": "{:.1%}", "CAGR": "{:.1%}", "Ann. volatility": "{:.1%}",
                                           "Max drawdown": "{:.1%}", "Beta vs Nifty": "{:.2f}",
                                           "Corr. with Asian Paints": "{:.2f}"}, na_rep="–"),
                         hide_index=True, width="stretch")
            fig = go.Figure()
            for i, row in tbl.iterrows():
                fig.add_trace(go.Scatter(x=[row["Ann. volatility"]], y=[row["CAGR"]], mode="markers+text",
                                         text=[row["Company"]], textposition="top center", name=row["Company"],
                                         marker=dict(size=13, color=ui.color(i))))
            fig.update_xaxes(title="Annualised volatility", tickformat=".0%")
            fig.update_yaxes(tickformat=".0%")
            ui.chart(ui.style(fig, "Risk vs return (CAGR)", 400, legend=False, hover="closest"), key="p_rr")
    if st.button("Load fundamentals snapshot", key="p_fund"):
        st.session_state["peer_fund"] = True
    if st.session_state.get("peer_fund"):
        f = peer_fund()
        st.dataframe(f.style.format({"Market Cap (₹ Cr)": "{:,.0f}", "P/E (TTM)": "{:.1f}", "P/B": "{:.1f}",
                                     "ROE": "{:.1%}", "Net Margin": "{:.1%}", "Operating Margin": "{:.1%}",
                                     "Revenue Growth (YoY)": "{:.1%}", "Debt/Equity (%)": "{:.1f}",
                                     "Dividend Yield": "{:.2f}"}, na_rep="–"), hide_index=True, width="stretch")
        st.caption("Yahoo reports dividend yield inconsistently (sometimes as %, sometimes as a fraction) – verify.")
