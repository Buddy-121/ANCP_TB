"""End-to-end test: ingest synthetic files with AUTO mapping, then render every page.

Run:  python tests/make_sample_data.py && python tests/test_generic_upload.py
Uses a temporary database so your real data is never touched.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["APP_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.sqlite")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src import database as db  # noqa: E402
from src import ingestion as ing  # noqa: E402
from src import validation as val  # noqa: E402
from src.config import DATASETS, UNIT_TO_CRORE  # noqa: E402

FILES = {"usdinr": "usd_inr.csv", "brent": "brent_europe.xlsx", "wti": "wti_us.csv", "ohlc": "asian_paints_ohlc.csv",
         "income": "income_statement.xlsx", "balance": "balance_sheet.csv", "cashflow": "cash_flow.csv",
         "ratios": "financial_ratios.xlsx", "brokerage_q": "brokerage_quarterly.xlsx",
         "brokerage_y": "brokerage_yearly.csv"}
SAMPLE = ROOT / "tests" / "sample_data"


def ingest_all():
    for cat, fn in FILES.items():
        kind = DATASETS[cat]["kind"]
        sheets = ing.read_any((SAMPLE / fn).read_bytes(), fn)
        table = ing.detect_header(list(sheets.values())[0])
        mapping = ing.suggest_mapping(table, kind)
        mult = None
        unit = ""
        if kind == "statement":
            unit = ing.unit_hint(table) if cat != "ratios" else list(UNIT_TO_CRORE)[-1]
            mult = UNIT_TO_CRORE.get(unit)
        long_df, stats = ing.transform(table, kind, mapping, mult)
        checks = val.validate(table.df, long_df, kind, stats)
        print(f"{cat:12s} unit={unit!s:10s} records={len(long_df):6d} periods={long_df['period'].nunique():5d} "
              f"status={val.overall_status(checks)}  mapping={ {k: v for k, v in mapping.items() if k != 'period_cols'} }")
        assert not long_df.empty, cat
        db.save_dataset(cat, long_df, "test", fn, unit, mapping)


def run_pages():
    from streamlit.testing.v1 import AppTest
    pages = ["executive", "macro", "stock", "financials", "ratios", "brokerage", "predictive", "advisory",
             "reports", "data_management"]
    failures = 0
    for p in pages:
        at = AppTest.from_file(str(ROOT / "views" / f"{p}.py"), default_timeout=180)
        at.run()
        errs = [e.value for e in at.exception]
        print(f"page {p:16s} exceptions={len(errs)} warnings={[w.value[:80] for w in at.warning]}")
        for e in errs:
            print("   ", e)
        failures += len(errs)
    return failures


if __name__ == "__main__":
    ingest_all()
    n = run_pages()
    print("FAILURES:", n)
    sys.exit(1 if n else 0)
