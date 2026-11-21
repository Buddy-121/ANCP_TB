"""Automated Research Reports – HTML / Markdown / Excel."""
import json

import streamlit as st
import streamlit.components.v1 as components

from src import data_access as da
from src import database as db
from src import insights
from src import reports as rp
from src import ui

ui.header("Automated Research Reports", "Generate a downloadable equity-research style report from the current data")
if not da.data_status()["Loaded"].any():
    ui.require(["income"])

SECTIONS = ["Executive summary", "Stock analysis", "Macroeconomic analysis", "Financial analysis",
            "Brokerage expectations", "Scenarios & forecasts", "AI recommendations"]
sections = st.multiselect("Sections", SECTIONS, default=SECTIONS, key="rep_sec")
hist = db.list_ai_reports()
adv = None
if "AI recommendations" in sections:
    if hist.empty:
        st.caption("No AI advisory generated yet – the AI section will be omitted. Generate one on the AI Strategic "
                   "Advisory page.")
    else:
        pick = st.selectbox("AI advisory to include", hist["id"].tolist(),
                            format_func=lambda i: f"#{i} – {hist.set_index('id').loc[i, 'ts']}", key="rep_ai")
        adv = json.loads(hist.set_index("id").loc[pick, "response_json"])

if st.button("Generate report", type="primary", key="rep_go"):
    with st.spinner("Building report…"):
        ctx = insights.build_context(include_forecasts="Scenarios & forecasts" in sections)
        st.session_state["rep_html"] = rp.build_html(ctx, adv, sections)
        st.session_state["rep_md"] = rp.build_markdown(ctx, adv)
    db.log("report", f"sections={len(sections)} ai={'yes' if adv else 'no'}")

if st.session_state.get("rep_html"):
    s = rp.stamp()
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download HTML report", st.session_state["rep_html"].encode(), f"asian_paints_report_{s}.html",
                       "text/html", key="rep_dl_h")
    c2.download_button("Download Markdown", st.session_state["rep_md"].encode(), f"asian_paints_report_{s}.md",
                       "text/markdown", key="rep_dl_m")
    c3.download_button("Download Excel data pack", rp.build_excel_pack(), f"asian_paints_data_{s}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="rep_dl_x")
    st.caption("For PDF: open the HTML report in your browser → Print → Save as PDF (print styles are included).")
    if hasattr(st, "iframe"):  # Streamlit ≥ 1.5x
        st.iframe(st.session_state["rep_html"], height=900)
    else:  # older Streamlit
        components.html(st.session_state["rep_html"], height=900, scrolling=True)
