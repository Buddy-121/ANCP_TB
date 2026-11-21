"""SQLite storage with versioned datasets.

Every upload / fetch / manual edit creates a NEW version. Older versions are never
deleted or overwritten; only the `is_current` flag moves, and only after the user
explicitly confirms in the UI.

Generic long-format table `records`:
    timeseries / ohlc : period = date, item = field (value/open/high/low/close/volume)
    statement         : period = period end date, item = line item, value
    brokerage         : period, item = metric, value = expected, value2 = actual
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    version INTEGER NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0,
    source TEXT,
    filename TEXT,
    unit TEXT,
    row_count INTEGER,
    period_start TEXT,
    period_end TEXT,
    mapping_json TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    dataset_id INTEGER NOT NULL,
    period TEXT NOT NULL,
    item TEXT NOT NULL,
    value REAL,
    value2 REAL,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id)
);
CREATE INDEX IF NOT EXISTS idx_records_ds ON records(dataset_id);
CREATE TABLE IF NOT EXISTS item_map (
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    item TEXT NOT NULL,
    updated_at TEXT,
    PRIMARY KEY(category, key)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS ai_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    model TEXT,
    context_json TEXT,
    response_json TEXT
);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect():
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


def log(action: str, detail: str = "") -> None:
    with connect() as con:
        con.execute("INSERT INTO audit_log(ts, action, detail) VALUES (?,?,?)", (_now(), action, detail))


def save_dataset(category: str, long_df: pd.DataFrame, source: str, filename: str = "",
                 unit: str = "", mapping: dict | None = None, notes: str = "",
                 make_current: bool = False) -> tuple[int, int]:
    """Store a new version. Returns (dataset_id, version).

    The first version of a category becomes current automatically. Later versions
    become current only if `make_current=True` (UI requires explicit confirmation).
    """
    df = long_df.copy()
    df["period"] = pd.to_datetime(df["period"]).dt.strftime("%Y-%m-%d")
    if "value2" not in df:
        df["value2"] = None
    with connect() as con:
        cur = con.execute("SELECT COALESCE(MAX(version),0), COUNT(*) FROM datasets WHERE category=?", (category,))
        max_v, n = cur.fetchone()
        version = max_v + 1
        current = 1 if (n == 0 or make_current) else 0
        if current:
            con.execute("UPDATE datasets SET is_current=0 WHERE category=?", (category,))
        cur = con.execute(
            """INSERT INTO datasets(category, version, is_current, source, filename, unit, row_count,
               period_start, period_end, mapping_json, notes, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (category, version, current, source, filename, unit, len(df),
             df["period"].min() if len(df) else None, df["period"].max() if len(df) else None,
             json.dumps(mapping or {}, default=str), notes, _now()))
        ds_id = cur.lastrowid
        rows = [(ds_id, r.period, str(r.item), None if pd.isna(r.value) else float(r.value),
                 None if pd.isna(r.value2) else float(r.value2))
                for r in df[["period", "item", "value", "value2"]].itertuples(index=False)]
        con.executemany("INSERT INTO records(dataset_id, period, item, value, value2) VALUES (?,?,?,?,?)", rows)
    log("save_dataset", f"{category} v{version} ({len(df)} rows) from {source} {filename}; current={bool(current)}")
    return ds_id, version


def list_versions(category: str | None = None) -> pd.DataFrame:
    q = "SELECT * FROM datasets"
    params: tuple = ()
    if category:
        q += " WHERE category=?"
        params = (category,)
    q += " ORDER BY category, version DESC"
    with connect() as con:
        return pd.read_sql_query(q, con, params=params)


def set_current(dataset_id: int) -> None:
    with connect() as con:
        cat = con.execute("SELECT category, version FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        if not cat:
            raise ValueError("Dataset version not found")
        con.execute("UPDATE datasets SET is_current=0 WHERE category=?", (cat[0],))
        con.execute("UPDATE datasets SET is_current=1 WHERE id=?", (dataset_id,))
    log("set_current", f"{cat[0]} -> v{cat[1]}")


def current_meta(category: str) -> dict | None:
    with connect() as con:
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT * FROM datasets WHERE category=? AND is_current=1", (category,)).fetchone()
        return dict(r) if r else None


def load_records(dataset_id: int) -> pd.DataFrame:
    with connect() as con:
        df = pd.read_sql_query("SELECT period, item, value, value2 FROM records WHERE dataset_id=?",
                               con, params=(dataset_id,))
    df["period"] = pd.to_datetime(df["period"])
    return df


def load_current(category: str) -> pd.DataFrame:
    meta = current_meta(category)
    if not meta:
        return pd.DataFrame(columns=["period", "item", "value", "value2"])
    return load_records(meta["id"])


def get_item_map(category: str) -> dict[str, str]:
    with connect() as con:
        rows = con.execute("SELECT key, item FROM item_map WHERE category=?", (category,)).fetchall()
    return {k: v for k, v in rows}


def set_item_map(category: str, key: str, item: str | None) -> None:
    with connect() as con:
        if item:
            con.execute("INSERT OR REPLACE INTO item_map(category, key, item, updated_at) VALUES (?,?,?,?)",
                        (category, key, item, _now()))
        else:
            con.execute("DELETE FROM item_map WHERE category=? AND key=?", (category, key))
    log("item_map", f"{category}.{key} -> {item}")


def audit_log(limit: int = 500) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query("SELECT ts, action, detail FROM audit_log ORDER BY id DESC LIMIT ?",
                                 con, params=(limit,))


def save_ai_report(model: str, context: dict, response: dict) -> int:
    with connect() as con:
        cur = con.execute("INSERT INTO ai_reports(ts, model, context_json, response_json) VALUES (?,?,?,?)",
                          (_now(), model, json.dumps(context, default=str), json.dumps(response, default=str)))
        rid = cur.lastrowid
    log("ai_report", f"id={rid} model={model}")
    return rid


def list_ai_reports() -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query("SELECT id, ts, model, context_json, response_json FROM ai_reports ORDER BY id DESC", con)


init_db()
