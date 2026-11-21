"""Shared UI components: theme-aware chart styling, KPI cards, formatting, guards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import data_access as da
from .config import DATASETS

# Categorical palette (fixed order, validated for CVD) – light / dark variants
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
GOOD, CRITICAL, WARNING = "#0ca30c", "#d03b3b", "#fab219"
DIVERGING = ["#d95926", "#f0efec", "#2a78d6"]  # orange ↔ neutral ↔ blue
DATA_PAGE = "views/data_management.py"


def is_dark() -> bool:
    """Best-effort detection of the active Streamlit theme (falls back to config base)."""
    try:
        t = st.context.theme.type
        if t in ("dark", "light"):
            return t == "dark"
    except Exception:  # noqa: BLE001
        pass
    return (st.get_option("theme.base") or "dark") == "dark"


def palette() -> list[str]:
    return SERIES_DARK if is_dark() else SERIES_LIGHT


def color(i: int) -> str:
    return palette()[i % 8]


def ink() -> dict:
    """Neutral inks used for reference lines/fills (readable on both surfaces)."""
    if is_dark():
        return {"primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"}
    return {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}


def style(fig: go.Figure, title: str | None = None, height: int = 420, y_title: str | None = None,
          legend: bool = True, hover: str = "x unified") -> go.Figure:
    """Theme-neutral styling: Streamlit's chart theme supplies fonts/text colours for light & dark."""
    grid, axis, muted = "rgba(137,135,129,0.22)", "rgba(137,135,129,0.45)", "#898781"
    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0, xanchor="left") if title else None,
        height=height, margin=dict(l=55, r=15, t=55 if title else 25, b=35),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        hovermode=hover, showlegend=legend, colorway=palette(),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=axis, tickfont=dict(color=muted), zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=grid, gridwidth=1, linecolor=axis, tickfont=dict(color=muted), zeroline=False,
                     automargin=True, title=dict(text=y_title) if y_title else None)
    return fig


def chart(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


# ------------------------------------------------------------------ formatting
def indian(n: float, dec: int = 0) -> str:
    """Indian digit grouping: 12,34,567."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "–"
    neg = n < 0
    n = abs(n)
    s = f"{n:.{dec}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("-" if neg else "") + whole + (f".{frac}" if frac else "")


def cr(v, dec: int = 0) -> str:
    return "–" if v is None or pd.isna(v) else f"₹{indian(v, dec)} Cr"


def pct(v, dec: int = 1, sign: bool = False) -> str:
    if v is None or pd.isna(v):
        return "–"
    return f"{v:+.{dec}%}" if sign else f"{v:.{dec}%}"


def num(v, dec: int = 2) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:,.{dec}f}"


def kpi(label: str, value: str, delta: str | None = None, help: str | None = None, inverse: bool = False,
        neutral: bool = False) -> None:
    st.metric(label, value, delta=delta, help=help, border=True,
              delta_color="off" if neutral else ("inverse" if inverse else "normal"))


def badge(kind: str) -> str:
    colors = {"FACT": "green", "FORECAST": "orange", "SCENARIO": "violet", "OPINION": "red", "ASSUMPTION": "gray"}
    return f":{colors.get(kind, 'gray')}-badge[{kind}]"


# ------------------------------------------------------------------ page helpers
def header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def require(categories: list[str], any_of: bool = False) -> list[str]:
    """Stop the page with a helpful message if required datasets are missing."""
    loaded = [c for c in categories if da.has(c)]
    missing = [c for c in categories if c not in loaded]
    ok = bool(loaded) if any_of else not missing
    if not ok:
        need = ", ".join(DATASETS[c]["label"] for c in (categories if any_of else missing))
        st.info(f"This module needs: **{need}**. Upload the file(s) or fetch market data in the Data Management Centre.")
        st.page_link(DATA_PAGE, label="Open Data Management Centre", icon=":material/database:")
        st.stop()
    return loaded


def date_range(index: pd.DatetimeIndex, key: str, default: str = "5Y") -> tuple[pd.Timestamp, pd.Timestamp]:
    """Preset + custom date filter rendered in one row above the charts."""
    lo, hi = pd.Timestamp(index.min()), pd.Timestamp(index.max())
    c1, c2 = st.columns([3, 2])
    presets = ["1Y", "3Y", "5Y", "10Y", "Max", "Custom"]
    choice = c1.segmented_control("Period", presets, default=default, key=f"{key}_preset") or default
    if choice == "Custom":
        d = c2.date_input("Custom range", (max(lo, hi - pd.DateOffset(years=3)).date(), hi.date()),
                          min_value=lo.date(), max_value=hi.date(), key=f"{key}_custom")
        if isinstance(d, tuple) and len(d) == 2:
            return pd.Timestamp(d[0]), pd.Timestamp(d[1])
        return lo, hi
    if choice == "Max":
        return lo, hi
    years = int(choice[:-1])
    return max(lo, hi - pd.DateOffset(years=years)), hi


def style_table(df: pd.DataFrame, pct_cols=(), num_cols=(), cr_cols=()):
    fmt = {c: (lambda v: pct(v)) for c in pct_cols if c in df}
    fmt.update({c: (lambda v: num(v)) for c in num_cols if c in df})
    fmt.update({c: (lambda v: indian(v, 0) if pd.notna(v) else "–") for c in cr_cols if c in df})
    return df.style.format(fmt, na_rep="–")


def inject_css() -> None:
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {border-radius: 10px; padding: 10px 14px;}
    div[data-testid="stMetricValue"] {font-size: 1.45rem; font-weight: 650;}
    div[data-testid="stMetricLabel"] p {font-size: 0.8rem; letter-spacing: .02em; text-transform: uppercase; opacity:.8}
    h2 {font-weight: 700; letter-spacing: -0.01em;}
    .block-container {padding-top: 2.2rem; max-width: 1500px;}
    </style>""", unsafe_allow_html=True)
