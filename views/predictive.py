"""Predictive Analytics & Scenario Modelling – forecasts, ML, cost scenarios, sensitivity, DCF."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import data_access as da
from src import forecasting as fc
from src import scenario as scn
from src import ui
from src.config import DATASETS, MACRO_SERIES, SHORT_NAMES

ui.header("Predictive Analytics & Scenario Modelling",
          "Statistical forecasts with honest backtests, input-cost scenarios, sensitivity and DCF valuation")
ui.require(["ohlc", *MACRO_SERIES, "income", "income_q"], any_of=True)

t1, t2, t3, t4, t5 = st.tabs(["Time-series forecast", "ML return model", "Cost scenario", "Sensitivity", "DCF valuation"])

# ================================================================== forecast
with t1:
    options = {}
    for k in ("ohlc", *MACRO_SERIES):
        if da.has(k):
            options[DATASETS[k]["label"]] = ("price", k)
    for cat, lbl in (("income", "annual"), ("income_q", "quarterly")):
        if da.has(cat):
            for key in ("revenue", "ebitda", "net_profit"):
                if da.key_series(cat, key) is not None:
                    options[f"{da.KEY_LABELS[key]} ({lbl})"] = ("stmt", (cat, key))
    if not options:
        st.info("No forecastable series loaded.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        name = c1.selectbox("Series", list(options), key="fc_s")
        typ, key = options[name]
        method = c2.selectbox("Model", fc.METHODS, index=2, key="fc_m")
        if typ == "price":
            s = an.resample(da.price_series(key), "ME", "last").dropna()
            freq, unit_lbl, hz_max = "ME", "months", 24
        else:
            scat, key = key
            s = da.key_series(scat, key).dropna()
            q = da.periodicity(s.index) == "quarterly"
            freq, unit_lbl, hz_max = ("QE-MAR" if q else "YE-MAR"), ("quarters" if q else "years"), 8
        horizon = c3.slider(f"Horizon ({unit_lbl})", 1, hz_max, min(6, hz_max), key="fc_h")
        use_log = c4.toggle("Model log values", value=bool((s > 0).all()), key="fc_log", disabled=not (s > 0).all())
        if len(s) < 10:
            st.warning(f"Only {len(s)} observations – forecasts on so little data are unreliable.")
        if len(s) >= 6:
            with st.spinner("Backtesting…"):
                bt = fc.walk_forward(s, method, n_test=12, log=use_log)
                f = fc.forecast(s, horizon, method, freq, log=use_log)
            fig = go.Figure()
            hist = s.tail(120)
            fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Actual", mode="lines", line=dict(width=2, color=ui.color(0))))
            fig.add_trace(go.Scatter(x=f.index, y=f["upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=f.index, y=f["lower"], mode="lines", fill="tonexty", fillcolor="rgba(235,104,52,0.15)",
                                     line=dict(width=0), name="95% interval"))
            fig.add_trace(go.Scatter(x=f.index, y=f["mean"], name="Forecast", mode="lines", line=dict(width=2, dash="dash", color=ui.color(1))))
            if "table" in bt:
                fig.add_trace(go.Scatter(x=bt["table"].index, y=bt["table"]["forecast"], name="Backtest (1-step)",
                                         mode="markers", marker=dict(size=8, color=ui.color(2))))
            cons_item = {"revenue": "Revenue", "ebitda": "EBITDA", "net_profit": "Net Income, GAAP"}.get(key)
            if typ == "stmt" and scat == "income" and cons_item and da.has("consensus_fy"):
                cw = da.statement_wide("consensus_fy")
                if cons_item in cw.index:
                    cs = cw.loc[cons_item].dropna()
                    fig.add_trace(go.Scatter(x=cs.index, y=cs.values, name="Bloomberg consensus (street)",
                                             mode="markers", marker=dict(size=12, symbol="diamond", color=ui.color(4))))
                    st.session_state["fc_cons"] = cs
            ui.chart(ui.style(fig, f"{name} – {method}", 430), key="fc_ch")
            st.markdown(ui.badge("FORECAST") + " Statistical projection – not a target or recommendation.")
            cs = st.session_state.pop("fc_cons", None)
            if cs is not None and len(cs):
                comp = ", ".join(f"{p:%b-%Y}: street ₹{v:,.0f} Cr" for p, v in cs.items())
                st.caption(f"Diamonds = Bloomberg consensus estimates ({comp}) – an independent benchmark for the model.")
            if "metrics" in bt:
                k = st.columns(4)
                with k[0]:
                    ui.kpi("Backtest MAPE", ui.pct(bt["metrics"]["MAPE"]))
                with k[1]:
                    ui.kpi("Naive MAPE", ui.pct(bt["naive_metrics"]["MAPE"]))
                with k[2]:
                    ui.kpi("Theil's U (RMSE ÷ naive)", ui.num(bt["theil_u"]), help="< 1 means the model beats naive")
                with k[3]:
                    ui.kpi("Beats naive?", "Yes" if bt["beats_naive"] else "No")
                if not bt["beats_naive"]:
                    st.warning("This model does not beat a naive 'no change' forecast out-of-sample. Treat the projection "
                               "as a scenario band, not a prediction.")
            ft = f.copy()
            ft.index = ft.index.strftime("%b-%Y")
            st.dataframe(ft.style.format("{:,.2f}"), width="stretch")
            with st.expander("Model assumptions & limitations"):
                st.markdown("""
- **Walk-forward backtest:** each point is forecast using only data available before it (expanding window).
- **Naive benchmark:** last observed value. Asset prices are close to a random walk, so naive is hard to beat.
- **ETS (damped trend):** additive errors, trend damped toward flat. **ARIMA(1,1,1):** one AR and one MA term on
  differenced data. **Linear trend:** OLS on time – ignores mean reversion and shocks.
- Intervals assume the model's error structure holds; regime changes (wars, OPEC decisions, pandemics) are not
  captured, so real uncertainty is wider than shown.
- Monthly price series use month-end closes.
""")

# ================================================================== ML
with t2:
    avail = {n: da.price_series(k) for k, n in [("ohlc", "AsianPaints"), ("brent", "Brent"), ("brent_fut", "BrentFut"),
                                                ("wti", "WTI"), ("usdinr", "USDINR")] if da.has(k)}
    if len(avail) < 2:
        st.info("Needs at least two price series (e.g. Asian Paints + Brent).")
    else:
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("Target – next month's log return of", list(avail), key="ml_t")
        lags = c2.slider("Feature lags (months)", 1, 6, 2, key="ml_l")
        splits = c3.slider("Time-series CV folds", 3, 10, 5, key="ml_s")
        if st.button("Train & backtest", type="primary", key="ml_go"):
            panel = an.aligned_panel(avail, "ME", "last")
            with st.spinner("Training…"):
                st.session_state["ml_res"] = fc.ml_return_model(panel, target, lags, splits)
        res = st.session_state.get("ml_res")
        if res:
            if "error" in res:
                st.warning(res["error"])
            else:
                st.markdown(ui.badge("FORECAST") + f" Out-of-sample results over {res['n']} monthly periods")
                st.dataframe(res["results"].style.format({"OOS R² vs train-mean": "{:.3f}", "Directional accuracy": "{:.1%}",
                                                          "Baseline directional (always up)": "{:.1%}", "RMSE": "{:.4f}",
                                                          "Test periods": "{:.0f}"}), width="stretch")
                best_r2 = res["results"]["OOS R² vs train-mean"].max()
                if best_r2 <= 0:
                    st.warning("Out-of-sample R² ≤ 0: the models do **not** predict next-month returns better than the "
                               "historical average. This is the expected result for a liquid large-cap stock and is itself "
                               "a finding: macro variables explain co-movement, but lagged macro data has little "
                               "predictive power for returns.")
                c1, c2 = st.columns(2)
                with c1:
                    imp = res["importance"].head(12)[::-1]
                    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ui.color(0)))
                    ui.chart(ui.style(fig, "Random-forest feature importance", 380, legend=False, hover="closest"), key="ml_imp")
                with c2:
                    o = res["oos"]["Ridge regression"]
                    fig = go.Figure([go.Scatter(x=o.index, y=o["actual"], name="Actual", line=dict(width=1.5, color=ui.color(0))),
                                     go.Scatter(x=o.index, y=o["predicted"], name="Ridge prediction",
                                                line=dict(width=1.5, color=ui.color(1)))])
                    fig.update_yaxes(tickformat=".0%")
                    ui.chart(ui.style(fig, "Out-of-sample predictions", 380), key="ml_oos")
                st.caption(f"Next-month prediction using features as of {res['features_asof']:%b-%Y}: " +
                           ", ".join(f"{k} {v:+.2%}" for k, v in res["next_period_pred"].items()))
                with st.expander("Leakage controls & limitations"):
                    st.markdown("""
- Features are log returns lagged by **at least one month** – no same-month information.
- **TimeSeriesSplit:** every test fold is strictly after its training data; the scaler is fitted inside each fold.
- Baseline = mean return of the **training** window only.
- Small samples (monthly data) make results noisy; several folds with different results indicate instability.
""")

# ================================================================== cost scenario
base = scn.base_financials() if da.has("income") else {}
base_ok = base and all(pd.notna(base.get(k)) for k in ("Revenue", "Material cost (COGS)", "EBITDA"))
with t3:
    if not base_ok:
        st.info("Needs an income statement with revenue, material cost and EBITDA inputs (see Line-item mapping).")
    else:
        st.markdown(f"{ui.badge('SCENARIO')} Base: **{base['basis']}** · Revenue {ui.cr(base['Revenue'])} · "
                    f"Material cost {ui.cr(base['Material cost (COGS)'])} · EBITDA {ui.cr(base['EBITDA'])}")
        link_df, link = an.crude_cost_link()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Market shocks**")
            crude = st.slider("Brent change (%)", -50, 100, 20, key="sc_c") / 100
            fx = st.slider("USD/INR change (%) (+ = rupee weaker)", -15, 25, 5, key="sc_f") / 100
            oth = st.slider("Employee & other cost inflation (%)", -10, 20, 0, key="sc_o") / 100
        with c2:
            st.markdown("**Cost structure assumptions**")
            share = st.slider("Crude-linked share of material cost (%)", 0, 90, 45, key="sc_s",
                              help="Monomers, solvents, additives derived from crude. Industry estimates vary ~40–55%.") / 100
            imp = st.slider("Other imported share of material cost (%)", 0, 50, 10, key="sc_i",
                            help="e.g. TiO₂ and pigments priced in USD") / 100
            if share + imp > 1:
                st.error("Shares exceed 100%.")
        with c3:
            st.markdown("**Management response**")
            price = st.slider("Price increase (%)", -5, 15, 0, key="sc_p") / 100
            elas = st.slider("Volume elasticity", -2.0, 0.0, -0.5, 0.1, key="sc_e",
                             help="% volume change per 1% price rise. Decorative paint demand is fairly inelastic.")
        sc = scn.CostScenario(crude, fx, share, imp, price, elas, oth)
        r = scn.run_cost_scenario(base, sc)
        k = st.columns(5)
        with k[0]:
            ui.kpi("Gross margin", ui.pct(r["new_gross_margin"]),
                   f"{(r['new_gross_margin'] - r['base_gross_margin']) * 100:+.1f} pp")
        with k[1]:
            ui.kpi("EBITDA margin", ui.pct(r["new_ebitda_margin"]),
                   f"{(r['new_ebitda_margin'] - r['base_ebitda_margin']) * 100:+.1f} pp")
        with k[2]:
            ui.kpi("Δ EBITDA", ui.cr(r["delta_ebitda"]))
        with k[3]:
            ui.kpi("≈ Δ PAT", ui.cr(r["approx_delta_pat"]), help=f"Δ EBITDA × (1 − {base['tax_rate']:.0%} tax)")
        with k[4]:
            ui.kpi("Break-even price hike", ui.pct(r["breakeven_price_increase"]),
                   help="Price increase that restores the base EBITDA margin, given the elasticity")
        steps = [("Base EBITDA", r["base_ebitda"], "absolute")]
        rev_eff = r["new_revenue"] - r["base_revenue"]
        mat_eff = -(r["new_cogs"] - r["base_cogs"])
        opex_eff = r["delta_ebitda"] - rev_eff - mat_eff
        steps += [("Revenue (price × volume)", rev_eff, "relative"), ("Material cost", mat_eff, "relative"),
                  ("Other costs", opex_eff, "relative"), ("Scenario EBITDA", r["new_ebitda"], "total")]
        fig = go.Figure(go.Waterfall(x=[s[0] for s in steps], y=[s[1] for s in steps], measure=[s[2] for s in steps],
                                     increasing=dict(marker_color=ui.GOOD), decreasing=dict(marker_color=ui.CRITICAL),
                                     totals=dict(marker_color=ui.color(0)), connector=dict(line=dict(color=ui.ink()["axis"]))))
        ui.chart(ui.style(fig, "EBITDA bridge (₹ Cr)", 380, legend=False, hover="closest"), key="sc_wf")
        if link.get("gm_pp_per_10pct_crude") is not None:
            model_pp = (scn.run_margin(base, scn.CostScenario(0.10, 0, share, imp, 0, elas, 0), 0)
                        - base["EBITDA"] / base["Revenue"]) * 100
            st.caption(f"Calibration check {ui.badge('FACT')}: history shows {link['gm_pp_per_10pct_crude']:+.2f} pp gross-margin "
                       f"change per +10% Brent (₹); this model gives {model_pp:+.2f} pp per +10% Brent with your cost shares. "
                       "History is smaller when price increases or hedging offset costs.")
        with st.expander("Model assumptions & limitations"):
            st.markdown("""
- Material cost index = crude share × (1+Δcrude)(1+ΔFX) + import share × (1+ΔFX) + domestic share.
- Shocks are applied to the full year instantly; in reality inventory (2–3 months) and contracts delay the impact.
- Volume elasticity is a single constant; competitor pricing response is not modelled.
- PAT impact ignores changes in depreciation, interest and other income.
""")

# ================================================================== sensitivity
with t4:
    if not base_ok:
        st.info("Needs the same inputs as the cost scenario.")
    else:
        sc = scn.CostScenario(crude_linked_share=st.session_state.get("sc_s", 45) / 100,
                              other_import_share=st.session_state.get("sc_i", 10) / 100,
                              pass_through_price=st.session_state.get("sc_p", 0) / 100,
                              volume_elasticity=st.session_state.get("sc_e", -0.5))
        cr_rng = np.arange(-0.3, 0.61, 0.1)
        fx_rng = np.arange(-0.05, 0.151, 0.025)
        g = scn.sensitivity_grid(base, sc, cr_rng, fx_rng)
        base_m = base["EBITDA"] / base["Revenue"]
        dev = (g - base_m) * 100
        lim = float(np.abs(dev.values).max())
        fig = go.Figure(go.Heatmap(z=dev.values, x=g.columns, y=g.index, zmin=-lim, zmax=lim,
                                   colorscale=[[0, ui.CRITICAL], [0.5, ui.ink()["grid"]], [1, ui.GOOD]],
                                   text=[[f"{v:.1%}" for v in row] for row in g.values], texttemplate="%{text}",
                                   xgap=2, ygap=2, hovertemplate="Brent %{x}, INR %{y}: margin %{text}<extra></extra>"))
        fig.update_xaxes(title="Brent change")
        fig.update_yaxes(title="USD/INR change")
        ui.chart(ui.style(fig, f"EBITDA margin sensitivity (base {base_m:.1%}; colour = change vs base)", 460,
                          legend=False, hover="closest"), key="sens")
        st.markdown(ui.badge("SCENARIO") + " Uses the cost-share, price and elasticity assumptions from the Cost scenario tab.")

        st.markdown("#### One-at-a-time sensitivity (tornado)")
        base_sc = scn.CostScenario(crude_linked_share=sc.crude_linked_share, other_import_share=sc.other_import_share,
                                   volume_elasticity=sc.volume_elasticity)
        tests = {"Brent ±20%": ("crude_change", 0.2), "USD/INR ±5%": ("fx_change", 0.05),
                 "Price ±3%": ("pass_through_price", 0.03), "Other costs ±5%": ("other_cost_inflation", 0.05)}
        rows = []
        for lbl, (fld, d) in tests.items():
            lo = scn.run_cost_scenario(base, scn.CostScenario(**{**base_sc.__dict__, fld: -d}))["new_ebitda_margin"]
            hi = scn.run_cost_scenario(base, scn.CostScenario(**{**base_sc.__dict__, fld: d}))["new_ebitda_margin"]
            rows.append({"Driver": lbl, "Low": (lo - base_m) * 100, "High": (hi - base_m) * 100})
        t = pd.DataFrame(rows).assign(span=lambda x: (x["High"] - x["Low"]).abs()).sort_values("span")
        fig = go.Figure([go.Bar(y=t["Driver"], x=t["Low"], orientation="h", name="Downside shock", marker_color=ui.color(1)),
                         go.Bar(y=t["Driver"], x=t["High"], orientation="h", name="Upside shock", marker_color=ui.color(0))])
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title="Change in EBITDA margin (pp)")
        ui.chart(ui.style(fig, "Tornado – EBITDA margin impact", 320, hover="closest"), key="tor")

# ================================================================== DCF
with t5:
    d = scn.dcf_defaults() if da.has("income") else {}
    if d.get("base_basis"):
        st.caption(f"Base revenue: {d['base_basis']}. Defaults use the last 3 fiscal years of your data.")
    if not d.get("base_revenue"):
        st.info("DCF needs an income statement with revenue.")
    else:
        st.markdown(ui.badge("SCENARIO") + " Valuation is driven entirely by the assumptions below – defaults are "
                    "derived from your data where possible and are editable.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Growth & margins**")
            hist_g = d.get("hist_rev_cagr")
            g1 = st.number_input("Stage-1 revenue growth (%)", -10.0, 40.0,
                                 round(float(np.clip((hist_g or 0.10) * 100, 0, 25)), 1), 0.5, key="dcf_g1") / 100
            y1 = st.number_input("Stage-1 years", 1, 10, 5, key="dcf_y1")
            g2 = st.number_input("Stage-2 revenue growth (%)", -5.0, 30.0, 8.0, 0.5, key="dcf_g2") / 100
            y2 = st.number_input("Stage-2 years", 0, 10, 5, key="dcf_y2")
            em = st.number_input("EBITDA margin (%)", 0.0, 60.0, round(d.get("ebitda_margin", 0.2) * 100, 1), 0.5,
                                 key="dcf_em") / 100
            if hist_g is not None:
                st.caption(f"Historical revenue CAGR (last 5 periods): {hist_g:.1%}")
        with c2:
            st.markdown("**Reinvestment & tax**")
            dep = st.number_input("Depreciation (% revenue)", 0.0, 20.0, round(d.get("dep_pct", 0.03) * 100, 2), 0.1,
                                  key="dcf_dep") / 100
            capex = st.number_input("Capex (% revenue)", 0.0, 30.0, round(d.get("capex_pct", 0.04) * 100, 2), 0.1,
                                    key="dcf_cap") / 100
            nwc = st.number_input("Working capital (% of incremental revenue)", -50.0, 100.0, 10.0, 1.0, key="dcf_nwc") / 100
            tax = st.number_input("Tax rate (%)", 0.0, 40.0, round(d.get("tax_rate", 0.25) * 100, 1), 0.5, key="dcf_tax") / 100
        with c3:
            st.markdown("**Discounting & equity bridge**")
            wacc = st.number_input("WACC (%)", 5.0, 20.0, 11.0, 0.25, key="dcf_w",
                                   help="Typical Indian large-cap: Rf ≈ 6.5–7% + β × ERP ≈ 6–7%") / 100
            tg = st.number_input("Terminal growth (%)", 0.0, 9.0, 5.0, 0.25, key="dcf_tg",
                                 help="Should not exceed long-run nominal GDP growth") / 100
            nc = st.number_input("Net cash (₹ Cr) (+ cash, − debt)", -1e6, 1e6, round(d.get("net_cash", 0.0), 0), 100.0,
                                 key="dcf_nc")
            sc_cap = d.get("share_capital_cr")
            fv = st.number_input("Face value per share (₹)", 0.1, 100.0, 1.0, 0.1, key="dcf_fv",
                                 help="Asian Paints' face value is ₹1 – verify")
            shares_default = (sc_cap / fv) if sc_cap else 95.92
            shares = st.number_input("Shares outstanding (crore)", 0.01, 10000.0, round(float(shares_default), 2), 0.01,
                                     key="dcf_sh", help="Default = equity share capital ÷ face value")
        inp = scn.DCFInputs(d["base_revenue"], g1, int(y1), g2, int(y2), em, dep, capex, nwc, tax, wacc, tg, nc, shares)
        res = scn.dcf(inp)
        if "error" in res:
            st.error(res["error"])
        else:
            price = da.price_series("ohlc").iloc[-1] if da.has("ohlc") else None
            k = st.columns(5)
            with k[0]:
                ui.kpi("Enterprise value", ui.cr(res["enterprise_value"]))
            with k[1]:
                ui.kpi("Equity value", ui.cr(res["equity_value"]))
            with k[2]:
                ui.kpi("Value per share", f"₹{res['per_share']:,.0f}")
            with k[3]:
                ui.kpi("Market price", f"₹{price:,.0f}" if price else "–",
                       ui.pct(res["per_share"] / price - 1, 1, True) + " implied" if price else None)
            with k[4]:
                ui.kpi("Terminal value % of EV", ui.pct(res["tv_share_of_ev"]),
                       help="Above ~75% means the valuation rests mostly on the terminal assumptions")
            tbl = res["table"]
            st.dataframe(tbl.style.format({c: "{:,.0f}" for c in tbl.columns if c != "Discount factor"} |
                                          {"Discount factor": "{:.3f}"}), width="stretch")
            st.markdown("**Value per share (₹) – WACC × terminal growth**")
            sens = scn.dcf_sensitivity(inp, np.arange(wacc - 0.02, wacc + 0.0201, 0.01), np.arange(tg - 0.01, tg + 0.0101, 0.005))
            st.dataframe(sens.style.format("{:,.0f}", na_rep="n/a"), width="stretch")
            with st.expander("DCF assumptions & limitations"):
                st.markdown("""
- FCFF = EBIT × (1 − tax) + depreciation − capex − Δ working capital; discounted at WACC, end-of-year convention.
- Terminal value = FCFF(final) × (1 + g) / (WACC − g) – very sensitive to WACC − g.
- Single-scenario DCF; real outcomes depend on market share, competitive intensity (new entrants) and raw-material cycles.
- Shares outstanding default is derived from share capital ÷ face value; check against the latest filing.
""")
