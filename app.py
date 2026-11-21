"""Asian Paints – AI-Powered Strategic Intelligence Platform.

Run from the project folder:   streamlit run app.py
"""
import streamlit as st  # type: ignore[import-not-found]

st.set_page_config(page_title="Asian Paints Strategic Intelligence", page_icon=":material/insights:",
                   layout="wide", initial_sidebar_state="expanded")

from src import bloomberg as bb  # noqa: E402  (after set_page_config)
from src import data_access as da  # noqa: E402
from src import database as db  # noqa: E402
from src import ui  # noqa: E402

db.init_db()
if "autoload_checked" not in st.session_state:
    st.session_state["autoload_checked"] = True
    with st.spinner("First run: loading the Bloomberg workbooks from data/input…"):
        loaded = bb.autoload_if_empty()
    if loaded:
        da.clear_cache()
        st.toast(f"Loaded {len(loaded)} datasets from data/input", icon=":material/check_circle:")
ui.inject_css()

pages = {
    "Overview": [
        st.Page("views/executive.py", title="Executive Dashboard", icon=":material/dashboard:", default=True),
    ],
    "Market & Macro": [
        st.Page("views/macro.py", title="Macroeconomic Intelligence", icon=":material/public:"),
        st.Page("views/stock.py", title="Stock Analysis", icon=":material/candlestick_chart:"),
    ],
    "Fundamentals": [
        st.Page("views/financials.py", title="Financial Statements", icon=":material/account_balance:"),
        st.Page("views/ratios.py", title="Financial Ratios", icon=":material/percent:"),
        st.Page("views/brokerage.py", title="Expectations vs Actuals", icon=":material/fact_check:"),
    ],
    "Intelligence": [
        st.Page("views/predictive.py", title="Predictive & Scenarios", icon=":material/query_stats:"),
        st.Page("views/advisory.py", title="AI Strategic Advisory", icon=":material/psychology:"),
        st.Page("views/reports.py", title="Research Reports", icon=":material/description:"),
    ],
    "Data": [
        st.Page("views/data_management.py", title="Data Management Centre", icon=":material/database:"),
    ],
}

with st.sidebar:
    st.markdown("### Asian Paints\n**Strategic Intelligence Platform**")
    st.caption("Theme: ⋮ menu → Settings → Light / Dark")

nav = st.navigation(pages)
nav.run()
