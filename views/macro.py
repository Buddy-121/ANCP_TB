"""Macroeconomic Intelligence – crude, FX, correlations, lagged effects, regression."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import data_access as da
from src import ui
from src.config import DATASETS, MACRO_SERIES, SHORT_NAMES

ui.header("Macroeconomic Intelligence", "How crude oil and the rupee move – and how they relate to Asian Paints' "
          "share price and margins")
loaded = ui.require(MACRO_SERIES, any_of=True)

series = {k: da.price_series(k) for k in loaded}
if da.has("ohlc"):
    series["stock"] = da.price_series("ohlc")
NAMES = {**SHORT_NAMES, "stock": "Asian Paints"}
all_idx = pd.DatetimeIndex(sorted(set().union(*[s.index for s in series.values()])))
start, end = ui.date_range(all_idx, "macro")
win = {k: s[(s.index >= start) & (s.index <= end)] for k, s in series.items()}

t1, t2, t3, t4 = st.tabs(["Trends", "Correlation", "Lagged effects & regression", "Crude → input costs"])

# ================================================================== trends
with t1:
    cols = st.columns(len(loaded))
    for i, k in enumerate(loaded):
        s = win[k]
        if s.empty:
            continue
        stt = an.price_stats(s) if len(s) > 30 else {}
        with cols[i]:
            ui.kpi(NAMES[k], f"{s.iloc[-1]:,.2f}", ui.pct(s.iloc[-1] / s.iloc[0] - 1, 1, True) + " in period",
                   help=DATASETS[k]["unit"], inverse=True)
            fig = go.Figure(go.Scatter(x=s.index, y=s.values, line=dict(width=2, color=ui.color(i)), name=NAMES[k]))
            fig.add_trace(go.Scatter(x=s.index, y=an.sma(series[k], 200).reindex(s.index), name="200-DMA",
                                     line=dict(width=1.2, color=ui.ink()["muted"], dash="dot")))
            ui.chart(ui.style(fig, f"{NAMES[k]} ({DATASETS[k]['unit']})", 300), key=f"m_tr_{k}")
            if stt:
                st.caption(f"Ann. volatility {ui.pct(stt['annual_volatility'])} · max drawdown {ui.pct(stt['max_drawdown'])}")
    c1, c2 = st.columns(2)
    other = "wti" if "wti" in win else ("brent_fut" if "brent_fut" in win else None)
    if "brent" in win and other:
        sp = (win["brent"] - win[other]).dropna()
        with c1:
            fig = go.Figure(go.Scatter(x=sp.index, y=sp.values, mode="lines", line=dict(width=1.5, color=ui.color(6)),
                                       name="Spread"))
            fig.add_hline(y=0, line=dict(color=ui.ink()["axis"], width=1))
            lbl = "Brent – WTI spread" if other == "wti" else "Brent spot – Dec-26 futures (COZ6): curve shape"
            ui.chart(ui.style(fig, f"{lbl} (USD/bbl)", 300, legend=False), key="m_spread")
            if other == "brent_fut":
                st.caption("Positive = spot above the Dec-26 contract (backwardation, tight near-term supply); negative = "
                           "contango. COZ6 is one contract, so its distance to expiry shrinks over time.")
    if "brent" in win and "usdinr" in win:
        bi = (win["brent"] * series["usdinr"].reindex(win["brent"].index).ffill()).dropna()
        with c2:
            fig = go.Figure(go.Scatter(x=bi.index, y=bi.values, line=dict(width=2, color=ui.color(1)), name="Brent ₹"))
            ui.chart(ui.style(fig, "Brent in rupees (₹/bbl) – the cost Asian Paints actually faces", 300, legend=False),
                     key="m_binr")
    st.caption(ui.badge("FACT") + " Brent in ₹ = Brent (USD) × USD/INR on the same day (FX forward-filled over holidays).")

    vol = pd.DataFrame({NAMES[k]: an.hist_vol(series[k], 63) for k in series}).loc[start:end]
    fig = go.Figure([go.Scatter(x=vol.index, y=vol[c], name=c, line=dict(width=1.8, color=ui.color(i)))
                     for i, c in enumerate(vol.columns)])
    fig.update_yaxes(tickformat=".0%")
    ui.chart(ui.style(fig, "Rolling 3-month annualised volatility", 320), key="m_vol")

# ================================================================== correlation
with t2:
    c1, c2, c3 = st.columns(3)
    freq = c1.selectbox("Frequency", ["Monthly", "Weekly"], key="c_freq")
    transform = c2.selectbox("Transform", ["Log returns (recommended)", "Price levels"], key="c_tr")
    method = c3.selectbox("Method", ["pearson", "spearman"], key="c_m")
    f = "ME" if freq == "Monthly" else "W-FRI"
    panel = an.aligned_panel({NAMES[k]: v for k, v in win.items()}, f, "last")
    data = an.to_changes(panel, "log" if transform.startswith("Log") else "level")
    if transform == "Price levels":
        st.warning("Level correlations between trending series are often spurious (shared trends, not shared "
                   "drivers). Use log returns for inference.")
    corr, pv, n = an.corr_matrix(data, method)
    if n < 12:
        st.warning(f"Only {n} overlapping observations – widen the date range.")
    text = [[f"{corr.iloc[i, j]:.2f}" + ("" if i == j or pd.isna(pv.iloc[i, j]) or pv.iloc[i, j] < 0.05 else "†")
             for j in range(len(corr))] for i in range(len(corr))]
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               colorscale=[[0, ui.DIVERGING[0]], [0.5, ui.ink()["grid"]], [1, ui.DIVERGING[2]]],
                               text=text, texttemplate="%{text}", hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
                               xgap=2, ygap=2))
    ui.chart(ui.style(fig, f"{method.title()} correlation – {freq.lower()} {transform.split(' (')[0].lower()} (n = {n})",
                      420, legend=False, hover="closest"), key="m_corr")
    st.caption(ui.badge("FACT") + " † = not significant at 5%. Correlation describes co-movement, not causation.")
    if "Asian Paints" in data and data.shape[1] > 1:
        drv = st.selectbox("Rolling correlation of Asian Paints with", [c for c in data.columns if c != "Asian Paints"],
                           key="c_drv")
        w = 12 if freq == "Monthly" else 26
        rc = an.rolling_corr(data[drv], data["Asian Paints"], w)
        fig = go.Figure(go.Scatter(x=rc.index, y=rc.values, line=dict(width=2, color=ui.color(0)), name="Rolling corr"))
        fig.add_hline(y=0, line=dict(color=ui.ink()["axis"], width=1))
        ui.chart(ui.style(fig, f"Rolling {w}-period correlation: {drv} vs Asian Paints", 300, legend=False), key="m_rc")
        st.caption("Unstable rolling correlations mean the relationship is regime-dependent – avoid relying on a single number.")

# ================================================================== lagged & regression
with t3:
    panel_m = an.aligned_panel({NAMES[k]: v for k, v in win.items()}, "ME", "last")
    rets = an.to_changes(panel_m, "log")
    if rets.shape[1] < 2:
        st.info("Load at least two series to analyse lagged effects.")
    else:
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("Target (monthly log return)", list(rets.columns),
                              index=list(rets.columns).index("Asian Paints") if "Asian Paints" in rets else 0, key="l_t")
        driver = c2.selectbox("Driver", [c for c in rets.columns if c != target], key="l_d")
        max_lag = c3.slider("Max lag (months)", 1, 12, 6, key="l_ml")
        lt = an.lagged_corr(rets[driver], rets[target], max_lag)
        if not lt.empty:
            colors = [ui.color(0) if s else ui.ink()["muted"] for s in lt["significant_5pct"]]
            fig = go.Figure(go.Bar(x=lt["lag"], y=lt["corr"], marker_color=colors,
                                   customdata=np.c_[lt["p_value"], lt["n"]],
                                   hovertemplate="Lag %{x}: r=%{y:.3f}<br>p=%{customdata[0]:.3f}, n=%{customdata[1]}<extra></extra>"))
            fig.add_hline(y=0, line=dict(color=ui.ink()["axis"], width=1))
            fig.update_xaxes(title="Months that driver leads target", dtick=1)
            ui.chart(ui.style(fig, f"Lagged correlation: {driver} (t−k) → {target} (t)", 330, legend=False, hover="closest"),
                     key="m_lag")
            st.caption(ui.badge("FACT") + " Blue = significant at 5%; grey = not significant. Lag k uses only driver "
                       "values from k months earlier – no look-ahead. With many lags tested, expect ~1 in 20 false positives.")

        st.markdown("#### Multiple linear regression")
        c1, c2 = st.columns([2, 1])
        drivers = c1.multiselect("Explanatory variables", [c for c in rets.columns if c != target],
                                 default=[c for c in rets.columns if c != target][:3], key="r_x")
        lag = c2.number_input("Lag drivers by (months)", 0, 12, 0, key="r_lag",
                              help="0 = contemporaneous (explanatory). ≥1 = predictive (uses only past values).")
        if drivers:
            X = rets[drivers].shift(int(lag))
            X.columns = [f"{c} (t−{int(lag)})" if lag else c for c in X.columns]
            res = an.ols_regression(rets[target], X)
            if "error" in res:
                st.warning(res["error"])
            else:
                k = st.columns(5)
                with k[0]:
                    ui.kpi("R²", f"{res['r2']:.3f}")
                with k[1]:
                    ui.kpi("Adj. R²", f"{res['adj_r2']:.3f}")
                with k[2]:
                    ui.kpi("Observations", f"{res['n']}")
                with k[3]:
                    ui.kpi("F-test p", f"{res['f_pvalue']:.3f}")
                with k[4]:
                    ui.kpi("Durbin–Watson", f"{res['durbin_watson']:.2f}", help="≈2 = no residual autocorrelation")
                st.dataframe(res["coefficients"].style.format("{:.4f}"), width="stretch")
                notes = []
                for name, row in res["coefficients"].drop(index="const").iterrows():
                    sig = "significant" if row["p_value"] < 0.05 else "not significant"
                    notes.append(f"- **{name}**: a 1% move is associated with a {row['coef']:.3f}% move in {target} "
                                 f"({sig}, p = {row['p_value']:.3f}).")
                st.markdown("\n".join(notes))
                if res["vif"] and max(res["vif"].values()) > 5:
                    st.warning("High multicollinearity (VIF > 5): " + ", ".join(f"{k} {v:.1f}" for k, v in res["vif"].items()
                                                                               if v > 5) + ". Brent and WTI usually move together – consider using one.")
                if res["jarque_bera_p"] < 0.05:
                    st.caption("Residuals are non-normal (Jarque–Bera p < 0.05); HAC standard errors are used, but "
                               "treat p-values as approximate.")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res["actual"].index, y=res["actual"], name="Actual",
                                         line=dict(width=2, color=ui.color(0))))
                fig.add_trace(go.Scatter(x=res["fitted"].index, y=res["fitted"], name="Fitted",
                                         line=dict(width=2, color=ui.color(1))))
                fig.update_yaxes(tickformat=".0%")
                ui.chart(ui.style(fig, f"{target}: actual vs fitted monthly returns", 320), key="m_fit")
                with st.expander("Model assumptions & limitations"):
                    st.markdown("""
- **Linear, stable relationship** across the whole window – rolling correlations show this often fails.
- **Returns, not levels**, to avoid spurious regression on trending prices.
- **HAC (Newey–West) standard errors** correct for autocorrelation and heteroskedasticity.
- **Omitted variables:** market index, rates, monsoon, housing demand and competition are not in the model,
  so coefficients can absorb their effects.
- Contemporaneous (lag 0) models explain; they cannot be used to forecast because the drivers are not
  known in advance.
""")

# ================================================================== crude → input costs
with t4:
    srcs = [k for k in ("income_q", "income") if da.has(k)]
    src = st.radio("Margin data", srcs, horizontal=True, key="m_csrc",
                   format_func={"income_q": "Quarterly P&L (more observations)", "income": "Annual statements"}.get) \
        if len(srcs) > 1 else (srcs[0] if srcs else "income")
    flags = an.margin_flags(src)
    excl = True
    if len(flags):
        n_out = int(flags["outlier_vs_neighbours"].sum())
        excl = st.toggle(f"Exclude data-quality outliers ({n_out} period(s) where the computed gross margin disagrees with the "
                         "vendor's reported figure by > 2 pp AND jumps > 5 pp vs adjacent periods)", value=True, key="m_excl")
        with st.expander("Flagged periods"):
            st.dataframe(flags.assign(period=flags["period"].dt.strftime("%b-%Y")).style.format(
                {"computed_pp": "{:.1f}", "reported_pp": "{:.1f}", "gap_pp": "{:+.1f}"}), hide_index=True)
    df, info = an.crude_cost_link(income_cat=src, exclude_flagged=excl)
    if df.empty or "error" in info and len(df) == 0:
        st.info(info.get("error", "Needs income statement, Brent and USD/INR."))
    else:
        if "error" in info:
            st.warning(info["error"])
        lbl = df.index.strftime("%b-%y")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(x=df["Brent (INR) avg"], y=df["Material cost % revenue"], mode="markers",
                                       text=lbl, marker=dict(size=10, color=ui.color(1)),
                                       hovertemplate="%{text}<br>Brent ₹%{x:,.0f}<br>Material %{y:.1%}<extra></extra>"))
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(title="Brent, ₹/bbl (period average)")
            ui.chart(ui.style(fig, "Material cost % of revenue vs Brent (₹)", 380, legend=False, hover="closest"),
                     key="m_sc")
        with c2:
            fig = go.Figure(go.Scatter(x=df["Brent (INR) avg"], y=df["Gross margin %"], mode="markers", text=lbl,
                                       marker=dict(size=10, color=ui.color(0)),
                                       hovertemplate="%{text}<br>Brent ₹%{x:,.0f}<br>GM %{y:.1%}<extra></extra>"))
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(title="Brent, ₹/bbl (period average)")
            ui.chart(ui.style(fig, "Gross margin vs Brent (₹)", 380, legend=False, hover="closest"), key="m_sc2")
        if "lag_table" in info:
            st.markdown(f"**Brent (₹) vs gross margin – {info.get('change_basis', 'changes')}, by lag** "
                        "(lag k = Brent change k periods earlier; p-values in this table are not autocorrelation-adjusted)")
            st.dataframe(info["lag_table"].style.format({"corr": "{:.3f}", "p_value": "{:.3f}"}), hide_index=True)
        if info.get("gm_pp_per_10pct_crude") is not None:
            lag_n = info["strongest_lag"]
            when = "in the same period" if lag_n == 0 else f"{lag_n} period(s) later"
            st.markdown(f"{ui.badge('FACT')} A **10% rise in Brent (₹)** has been associated with a "
                        f"**{info['gm_pp_per_10pct_crude']:+.2f} pp** change in gross margin {when} "
                        f"(R² {info['slope_r2']:.2f}, HAC p = {info['slope_p']:.3f}, n = {info.get('slope_n', info['n_periods'])}, "
                        f"{info.get('change_basis', '')}).")
        df = df.copy()
        df.index = df.index.strftime("%b-%Y")
        st.dataframe(df.style.format({"Brent (USD) avg": "{:.1f}", "USD/INR avg": "{:.2f}", "Brent (INR) avg": "{:,.0f}",
                                      "Material cost % revenue": "{:.1%}", "Gross margin %": "{:.1%}"}), width="stretch")
        st.caption("Quarters are Indian fiscal quarters (Q1 = Apr–Jun). Why a lag? Paint makers hold 2–3 months of raw-material inventory and take price increases with a "
                   "delay, so crude moves typically hit margins one or two quarters later. Crude is a proxy: the actual "
                   "basket (monomers, solvents, TiO₂, pigments) is only partly crude-linked.")
