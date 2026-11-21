"""Data Management Centre: upload, map, validate, version, fetch, edit, export."""
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import bloomberg as bb
from src import data_access as da
from src import database as db
from src import ingestion as ing
from src import market_data as md
from src import reports as rp
from src import ui
from src import validation as val
from src.config import DATASETS, DB_PATH, INPUT_DIR, RAW_DIR, UNIT_TO_CRORE, UPLOADABLE

ui.header("Data Management Centre",
          "Upload, map, validate and version every dataset. History is never overwritten – each change creates a new version.")

status = da.data_status()
st.dataframe(status.drop(columns=["Key"]), hide_index=True, width="stretch",
             column_config={"Loaded": st.column_config.CheckboxColumn()})

LABELS = {k: v["label"] for k, v in DATASETS.items()}
tabs = st.tabs(["Bloomberg import", "Upload & map (any file)", "Fetch market data", "Manual edit",
                "Versions & history", "Line-item mapping", "Export", "Audit log"])


def _sel(label, options, default, key, allow_none=False):
    opts = ([None] if allow_none else []) + list(options)
    idx = opts.index(default) if default in opts else 0
    return st.selectbox(label, opts, index=idx, key=key, format_func=lambda x: "— none —" if x is None else str(x))


def _confirm_current(cat: str, key: str) -> bool:
    if da.has(cat):
        return st.checkbox("Make this the **current** version used by all analyses "
                           "(the existing version stays in history and can be restored)", key=key)
    st.caption("No existing data for this dataset – this upload becomes the current version.")
    return True


def _preview_chart(long_df: pd.DataFrame, kind: str):
    if kind in ("timeseries", "ohlc"):
        f = "value" if kind == "timeseries" else "close"
        s = long_df[long_df["item"] == f].set_index("period")["value"].sort_index()
        fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(width=2, color=ui.color(0)),
                                   name=f))
        ui.chart(ui.style(fig, "Preview", 260, legend=False), key=f"prev_{kind}")


# ============================================================ Upload & map
def _upload_tab():
    c1, c2 = st.columns([1, 2])
    cat = c1.selectbox("Dataset", UPLOADABLE, format_func=LABELS.get, key="up_cat")
    kind = DATASETS[cat]["kind"]
    up = c2.file_uploader("CSV or Excel file", type=["csv", "txt", "xlsx", "xls", "xlsm"], key="up_file")
    if up is not None:
        try:
            sheets = ing.read_any(up.getvalue(), up.name)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read file: {e}")
            return
        if not sheets:
            st.error("The file contains no data.")
            return
        sheet = st.selectbox("Sheet", list(sheets), key="up_sheet") if len(sheets) > 1 else list(sheets)[0]
        raw = sheets[sheet]
        auto = ing.detect_header(raw)
        hc1, hc2 = st.columns([1, 3])
        hdr = hc1.number_input("Header row (0-based, after removing blank rows)", 0, max(0, len(raw) - 1),
                               value=auto.header_row, key="up_hdr")
        table = ing.detect_header(raw, header_override=int(hdr))
        if table.preamble:
            hc2.caption("Text above header: " + " | ".join(table.preamble)[:400])
        with st.expander("Raw preview (after header detection)", expanded=False):
            st.dataframe(table.df.head(20), width="stretch")

        st.markdown("#### Column mapping")
        st.caption("Auto-detected suggestions are pre-selected. Override anything that is wrong.")
        sug = ing.suggest_mapping(table, kind)
        cols = list(table.df.columns)
        mapping: dict = {}
        multiplier = None
        unit_label = DATASETS[cat].get("unit", "")
        if kind == "timeseries":
            m1, m2 = st.columns(2)
            mapping["date_col"] = _sel("Date column", cols, sug["date_col"], "m_date")
            mapping["value_col"] = _sel("Value column", cols, sug["value_col"], "m_val")
        elif kind == "ohlc":
            mc = st.columns(6)
            for i, f in enumerate(["date", "open", "high", "low", "close", "volume"]):
                with mc[i]:
                    mapping[f"{f}_col"] = _sel(f.title(), cols, sug.get(f"{f}_col"), f"m_{f}",
                                               allow_none=f in ("open", "high", "low", "volume"))
        elif kind == "statement":
            orient = st.radio("Layout", ["items_in_rows", "periods_in_rows"],
                              index=0 if sug["orientation"] == "items_in_rows" else 1, horizontal=True,
                              format_func={"items_in_rows": "Line items in rows, periods in columns",
                                           "periods_in_rows": "Periods in rows, line items in columns"}.get,
                              key="m_orient")
            mapping["orientation"] = orient
            if orient == "items_in_rows":
                mapping["item_col"] = _sel("Line-item column", cols, sug.get("item_col", cols[0]), "m_item")
                default_pc = sug.get("period_cols") or [c for c in cols if ing.parse_period_value(c) is not None]
                mapping["period_cols"] = st.multiselect("Period columns", [c for c in cols if c != mapping["item_col"]],
                                                        default=[c for c in default_pc if c != mapping["item_col"]],
                                                        key="m_pcols")
                if mapping["period_cols"]:
                    st.caption("Parsed periods: " + ", ".join(
                        f"{c} → {ing.parse_period_value(c):%b-%Y}" if ing.parse_period_value(c) is not None else f"{c} → ✗"
                        for c in mapping["period_cols"][:12]))
            else:
                mapping["period_col"] = _sel("Period/date column", cols, sug.get("period_col"), "m_pcol")
                mapping["item_cols"] = st.multiselect("Line-item columns", [c for c in cols if c != mapping["period_col"]],
                                                      default=[c for c in (sug.get("item_cols") or []) if c != mapping["period_col"]],
                                                      key="m_icols")
            hint = ing.unit_hint(table)
            unit_opts = list(UNIT_TO_CRORE)
            default_unit = unit_opts[-1] if cat == "ratios" else hint
            unit_label = st.selectbox(
                "Source unit of the figures (required – the platform standardises to ₹ Crore)", unit_opts,
                index=unit_opts.index(default_unit) if default_unit in unit_opts else None,
                placeholder="Select the unit used in this file", key="m_unit",
                help=f"Auto-detected hint: {hint or 'none found'}. Never assumed silently.")
            multiplier = UNIT_TO_CRORE.get(unit_label) if unit_label else None
            if unit_label is None:
                st.warning("Select the source unit to continue.")
        elif kind == "brokerage":
            fmt = st.radio("Layout", ["long", "wide"], index=0 if sug["format"] == "long" else 1, horizontal=True,
                           format_func={"long": "One row per period & metric (Metric | Expected | Actual)",
                                        "wide": "Paired columns (e.g. Revenue Expected | Revenue Actual)"}.get,
                           key="m_bfmt")
            mapping["format"] = fmt
            mapping["transposed"] = sug.get("transposed", False)
            work_cols = cols
            if mapping["transposed"]:
                pcs = [c for c in cols[1:] if ing.parse_period_value(c) is not None]
                work_cols = list(ing.brokerage_transpose(table.df, cols[0], pcs).columns)
                st.caption("Periods were detected as columns – the table is transposed automatically.")
            mapping["period_col"] = _sel("Period column", work_cols, sug.get("period_col"), "m_bper")
            if fmt == "long":
                b1, b2, b3 = st.columns(3)
                with b1:
                    mapping["metric_col"] = _sel("Metric column", work_cols, sug.get("metric_col"), "m_bmet")
                with b2:
                    mapping["expected_col"] = _sel("Expected column", work_cols, sug.get("expected_col"), "m_bexp")
                with b3:
                    mapping["actual_col"] = _sel("Actual column", work_cols, sug.get("actual_col"), "m_bact")
            else:
                pairs_df = pd.DataFrame(sug.get("pairs") or [{"metric": "", "expected_col": None, "actual_col": None}])
                pairs_df = st.data_editor(
                    pairs_df, num_rows="dynamic", key="m_pairs", width="stretch",
                    column_config={"metric": st.column_config.TextColumn("Metric name"),
                                   "expected_col": st.column_config.SelectboxColumn("Expected column", options=work_cols),
                                   "actual_col": st.column_config.SelectboxColumn("Actual column", options=work_cols)})
                mapping["pairs"] = [r for r in pairs_df.to_dict("records")
                                    if r.get("metric") and r.get("expected_col") and r.get("actual_col")]
            unit_label = st.text_input("Unit note (e.g. ₹ Crore, ₹ per share)", value="", key="m_bunit")

        ready = not (kind == "statement" and unit_label is None)
        if ready:
            try:
                long_df, stats = ing.transform(table, kind, mapping, multiplier)
            except Exception as e:  # noqa: BLE001
                st.error(f"Mapping could not be applied: {type(e).__name__}: {e}. Adjust the mapping above.")
                return
            checks = val.validate(table.df, long_df, kind, stats)
            overall = val.overall_status(checks)
            st.markdown(f"#### Validation · **{overall}**")
            st.dataframe(checks, hide_index=True, width="stretch")
            v1, v2 = st.columns([1, 1])
            with v1:
                st.markdown("**Standardised records (preview)**")
                st.dataframe(long_df.head(200), hide_index=True, width="stretch", height=260)
            with v2:
                _preview_chart(long_df, kind)
            notes = st.text_input("Notes for this version (optional)", key="up_notes")
            make_current = _confirm_current(cat, "up_current")
            disabled = long_df.empty or overall == "FAIL"
            if st.button("Save to database", type="primary", disabled=disabled, key="up_save"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                (RAW_DIR / f"{cat}_{ts}_{up.name}").write_bytes(up.getvalue())
                ds_id, v = db.save_dataset(cat, long_df, "upload", up.name, unit_label or "",
                                           {**mapping, "sheet": sheet, "header_row": int(hdr)}, notes, make_current)
                da.clear_cache()
                st.success(f"Saved {LABELS[cat]} as version {v}" + (" (current)." if make_current else
                           " (stored in history, not current)."))
                st.rerun()


with tabs[1]:
    _upload_tab()


# ============================================================ Bloomberg import
def _bloomberg_tab():
    st.markdown("Import Bloomberg terminal exports in their original layout: price sheets (`Security / PX_LAST …`), "
                "standardized statements (`In Millions of INR`, FY columns) and BEst *Company Financial (Single Period)* "
                "consensus sheets. Sheets are recognised by content, not by name. INR millions → **₹ Crore**; "
                "per-share values, %, ratios and share counts are not converted.")
    bundled = sorted(p for p in INPUT_DIR.glob("*.xls*") if not p.name.startswith("~$")) if INPUT_DIR.exists() else []
    src_choice = st.radio("Source", ["Files in data/input folder", "Upload workbooks"], horizontal=True, key="bb_src",
                          index=0 if bundled else 1)
    files: dict[str, bytes] = {}
    if src_choice.startswith("Files"):
        if not bundled:
            st.info(f"No Excel files in `{INPUT_DIR}`. Copy CP_DATA1.xlsx / CPD.xlsx there or upload them.")
        else:
            st.caption("Found: " + ", ".join(f"`{p.name}`" for p in bundled))
            files = {p.name: p.read_bytes() for p in bundled}
    else:
        ups = st.file_uploader("Bloomberg workbooks", type=["xlsx", "xls"], accept_multiple_files=True, key="bb_up")
        files = {u.name: u.getvalue() for u in ups or []}
    if not files:
        return
    if st.button("Analyse workbooks", key="bb_parse", type="primary"):
        with st.spinner("Reading sheets…"):
            try:
                st.session_state["bb_parsed"] = bb.load_workbooks(files) + (", ".join(files),)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the workbooks: {type(e).__name__}: {e}")
                return
    parsed = st.session_state.get("bb_parsed")
    if not parsed:
        return
    datasets, report, fnames = parsed
    st.markdown("#### What was found")
    st.dataframe(pd.DataFrame(report), hide_index=True, width="stretch")
    if "brent_fut" in datasets:
        st.info("The sheet with security **COZ6 Comdty** is the ICE *Brent* December-2026 futures contract – not WTI. "
                "It is stored as its own series (Brent Futures – COZ6). WTI stays empty until a WTI file (e.g. CL1 Comdty) "
                "is uploaded.")
    rows = []
    for key, df in datasets.items():
        meta = db.current_meta(key)
        rows.append({"Dataset": LABELS.get(key, key), "key": key, "Records": len(df),
                     "From": f"{df['period'].min():%d-%b-%Y}", "To": f"{df['period'].max():%d-%b-%Y}",
                     "Items": df["item"].nunique(), "Currently stored": f"v{meta['version']}" if meta else "—",
                     "Make current": True})
    plan = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", key="bb_plan",
                          disabled=["Dataset", "key", "Records", "From", "To", "Items", "Currently stored"],
                          column_config={"key": None,
                                         "Make current": st.column_config.CheckboxColumn(
                                             help="Untick to keep the existing version as current (new data is still "
                                                  "stored in history)")})
    existing = [r["Dataset"] for r in rows if r["Currently stored"] != "—"]
    confirm = True
    if existing:
        confirm = st.checkbox(f"I confirm: {len(existing)} dataset(s) already exist. Ticked rows will become the "
                              "current version; previous versions remain in history.", key="bb_conf")
    with st.expander("Preview a dataset"):
        pk = st.selectbox("Dataset", list(datasets), format_func=lambda k: LABELS.get(k, k), key="bb_prev")
        d = datasets[pk]
        if DATASETS.get(pk, {}).get("kind") == "statement":
            w = d.pivot_table(index="item", columns="period", values="value", sort=False)
            w.columns = [c.strftime("%b-%Y") for c in w.columns]
            st.dataframe(w.style.format("{:,.2f}", na_rep="–"), width="stretch", height=380)
        else:
            st.dataframe(d.head(500), hide_index=True, width="stretch", height=300)
    if st.button("Save all to database", type="primary", disabled=not confirm, key="bb_save"):
        cur = dict(zip(plan["key"], plan["Make current"]))
        res = bb.save_all(datasets, fnames, cur)
        da.clear_cache()
        st.session_state.pop("bb_parsed", None)
        st.success(f"Saved {len(res)} datasets.")
        st.rerun()


with tabs[0]:
    _bloomberg_tab()

# ============================================================ Fetch market data
with tabs[2]:
    st.caption("Source: Yahoo Finance via yfinance (free, unofficial). Brent = BZ=F, WTI = CL=F front-month futures; "
               "USD/INR = INR=X; Asian Paints = ASIANPAINT.NS.")
    mkt = [k for k, v in DATASETS.items() if v.get("ticker")]
    f1, f2, f3 = st.columns([2, 1, 1])
    sel = f1.multiselect("Datasets", mkt, default=mkt, format_func=LABELS.get, key="f_sel")
    mode = f2.radio("Mode", ["Incremental (new dates only)", "Full history from date"], key="f_mode")
    start = f3.date_input("Start date", date(2010, 1, 1), key="f_start",
                          disabled=mode.startswith("Incremental"))
    overwrite = st.checkbox("Overwrite existing dates where fetched values differ (otherwise history is kept)",
                            key="f_over")
    if st.button("Fetch", type="primary", key="f_go", disabled=not sel):
        results = {}
        for k in sel:
            meta = DATASETS[k]
            with st.spinner(f"Fetching {meta['label']}…"):
                try:
                    if mode.startswith("Incremental") and da.has(k):
                        last = da.records(k)["period"].max()
                        fstart = (last - pd.Timedelta(days=10)).date()
                    else:
                        fstart = start
                    raw = md.fetch_history(meta["ticker"], start=fstart, end=date.today() + pd.Timedelta(days=1))
                    new = md.to_long(raw, meta["kind"])
                    merged, stt = ing.merge_incremental(da.records(k), new, overwrite)
                    results[k] = {"merged": merged, "stats": stt, "fetched": len(raw)}
                except Exception as e:  # noqa: BLE001
                    results[k] = {"error": str(e)}
        st.session_state["fetch_results"] = results

    res = st.session_state.get("fetch_results")
    if res:
        rows = []
        for k, r in res.items():
            if "error" in r:
                rows.append({"Dataset": LABELS[k], "Status": "Error", "Detail": r["error"]})
            else:
                s = r["stats"]
                rows.append({"Dataset": LABELS[k], "Status": "Ready", "Detail":
                             f"{r['fetched']} rows fetched · {s['added']} new records · {s['conflicts']} conflicts · "
                             f"{s['unchanged']} unchanged"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for k, r in res.items():
            if "stats" in r and r["stats"]["conflicts"]:
                with st.expander(f"Conflicts – {LABELS[k]}"):
                    st.dataframe(r["stats"]["conflict_rows"], hide_index=True)
        ready = [k for k, r in res.items() if "merged" in r and (r["stats"]["added"] or (overwrite and r["stats"]["conflicts"]))]
        if ready:
            cur = st.checkbox("Make the merged data the current version for these datasets", value=True, key="f_cur")
            if st.button(f"Save {len(ready)} dataset(s) as new versions", key="f_save"):
                for k in ready:
                    db.save_dataset(k, res[k]["merged"], "yfinance", DATASETS[k]["ticker"], DATASETS[k]["unit"],
                                    {"ticker": DATASETS[k]["ticker"], "mode": mode}, "API fetch", cur)
                da.clear_cache()
                st.session_state.pop("fetch_results")
                st.success("Saved.")
                st.rerun()
        else:
            st.info("Nothing new to save.")

# ============================================================ Manual edit
with tabs[3]:
    cat_e = st.selectbox("Dataset", list(LABELS), format_func=LABELS.get, key="e_cat")
    kind_e = DATASETS[cat_e]["kind"]
    if not da.has(cat_e):
        st.info("No current data for this dataset. Upload it first.")
    else:
        r = da.records(cat_e)
        st.caption("Edit cells or add rows, then save. A new version is created; the previous one is preserved.")
        if kind_e == "timeseries":
            w = r[["period", "value"]].sort_values("period").reset_index(drop=True)
        elif kind_e == "ohlc":
            w = r.pivot_table(index="period", columns="item", values="value").reset_index()
        elif kind_e == "statement":
            w = r.pivot_table(index="item", columns="period", values="value", sort=False)
            w.columns = [c.strftime("%Y-%m-%d") for c in w.columns]
            w = w.reset_index()
        else:
            w = r.rename(columns={"item": "metric", "value": "expected", "value2": "actual"})[
                ["period", "metric", "expected", "actual"]]
        edited = st.data_editor(w, num_rows="dynamic", width="stretch", height=420, key=f"e_ed_{cat_e}")
        enote = st.text_input("Reason for change (recorded in audit log)", key="e_note")
        cur_e = st.checkbox("Make edited data the current version", key="e_cur")
        if st.button("Save edits as new version", type="primary", key="e_save", disabled=not enote):
            if kind_e == "timeseries":
                long_e = edited.assign(item="value")
            elif kind_e == "ohlc":
                long_e = edited.melt(id_vars="period", var_name="item", value_name="value")
            elif kind_e == "statement":
                long_e = edited.melt(id_vars="item", var_name="period", value_name="value")
            else:
                long_e = edited.rename(columns={"metric": "item", "expected": "value", "actual": "value2"})
            long_e["period"] = pd.to_datetime(long_e["period"], errors="coerce")
            long_e = long_e.dropna(subset=["period", "item"])
            meta = db.current_meta(cat_e)
            db.save_dataset(cat_e, long_e, "manual edit", meta.get("filename") or "", meta.get("unit") or "",
                            {"edited_from_version": meta["version"]}, enote, cur_e)
            da.clear_cache()
            st.success("Edits saved as a new version.")
            st.rerun()
        if not enote:
            st.caption("Enter a reason to enable saving.")

# ============================================================ Versions
with tabs[4]:
    v = db.list_versions()
    if v.empty:
        st.info("No datasets stored yet.")
    else:
        v["dataset"] = v["category"].map(LABELS)
        st.dataframe(v[["id", "dataset", "version", "is_current", "source", "filename", "unit", "row_count",
                        "period_start", "period_end", "notes", "created_at"]], hide_index=True, width="stretch",
                     column_config={"is_current": st.column_config.CheckboxColumn("current")})
        vid = st.selectbox("Inspect version (id)", v["id"].tolist(),
                           format_func=lambda i: f"#{i} – {LABELS[v.set_index('id').loc[i, 'category']]} v{v.set_index('id').loc[i, 'version']}",
                           key="v_sel")
        recs = db.load_records(int(vid))
        st.dataframe(recs.head(300), hide_index=True, width="stretch", height=250)
        c1, c2 = st.columns(2)
        c1.download_button("Download this version (CSV)", recs.to_csv(index=False).encode(), f"version_{vid}.csv",
                           "text/csv", key="v_dl")
        is_cur = bool(v.set_index("id").loc[vid, "is_current"])
        if not is_cur:
            conf = c2.checkbox("I confirm switching the current version", key="v_conf")
            if c2.button("Make this version current", disabled=not conf, key="v_set"):
                db.set_current(int(vid))
                da.clear_cache()
                st.success("Current version updated.")
                st.rerun()
        else:
            c2.caption("This is the current version.")

# ============================================================ Line-item mapping
with tabs[5]:
    st.caption("Ratios need to know which line item is 'Revenue', 'Raw material cost', etc. Items are matched "
               "automatically by name; override any mismatch here.")
    cat_m = st.selectbox("Statement", ["income", "income_q", "balance", "cashflow"], format_func=LABELS.get, key="lm_cat")
    if not da.has(cat_m):
        st.info("Upload this statement first.")
    else:
        items = list(da.statement_wide(cat_m).index)
        resolved = da.resolved_items(cat_m)
        overrides = db.get_item_map(cat_m)
        new_sel = {}
        cols = st.columns(3)
        for i, (key, item) in enumerate(resolved.items()):
            with cols[i % 3]:
                opts = [None] + items
                new_sel[key] = st.selectbox(
                    da.KEY_LABELS.get(key, key) + (" ✎" if key in overrides else ""), opts,
                    index=opts.index(item) if item in opts else 0, key=f"lm_{cat_m}_{key}",
                    format_func=lambda x: "— not available —" if x is None else x)
        if st.button("Save mapping", type="primary", key="lm_save"):
            for key, item in new_sel.items():
                if item != resolved.get(key) or key in overrides:
                    db.set_item_map(cat_m, key, item)
            da.clear_cache()
            st.success("Mapping saved.")
            st.rerun()

# ============================================================ Export
with tabs[6]:
    st.download_button("Download Excel data pack (all current data + computed ratios)", rp.build_excel_pack(),
                       f"asian_paints_data_{rp.stamp()}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="x_pack")
    cat_x = st.selectbox("Single dataset (CSV, standardised long format)", list(LABELS), format_func=LABELS.get, key="x_cat")
    st.download_button("Download CSV", da.records(cat_x).to_csv(index=False).encode(), f"{cat_x}_{rp.stamp()}.csv",
                       "text/csv", key="x_csv", disabled=not da.has(cat_x))
    if DB_PATH.exists():
        st.download_button("Download SQLite database (full history)", DB_PATH.read_bytes(), "platform.sqlite",
                           "application/x-sqlite3", key="x_db")

# ============================================================ Audit
with tabs[7]:
    st.dataframe(db.audit_log(), hide_index=True, width="stretch", height=450)
