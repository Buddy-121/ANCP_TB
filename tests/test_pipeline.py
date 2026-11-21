"""End-to-end test with the real Bloomberg workbooks in data/input.

Run:  python tests/test_pipeline.py
Uses a temporary database – your real database is never touched.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["APP_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.sqlite")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src import bloomberg as bb  # noqa: E402
from src.config import INPUT_DIR  # noqa: E402

PAGES = ["executive", "macro", "stock", "financials", "ratios", "brokerage", "predictive", "advisory", "reports",
         "data_management"]


def load():
    files = {p.name: p.read_bytes() for p in sorted(INPUT_DIR.glob("*.xls*"))}
    assert files, f"Put the Bloomberg workbooks in {INPUT_DIR}"
    datasets, report = bb.load_workbooks(files)
    for r in report:
        print(f"  {r['Sheet'][:26]:26s} → {str(r['Dataset']):18s} {r['Records']:>6}  {r['Detail'][:80]}")
    bb.save_all(datasets, ", ".join(files))


def run_pages() -> int:
    from streamlit.testing.v1 import AppTest
    failures = 0
    for p in PAGES:
        at = AppTest.from_file(str(ROOT / "views" / f"{p}.py"), default_timeout=240)
        at.run()
        errs = [e.value for e in at.exception]
        print(f"page {p:16s} exceptions={len(errs)} warnings={[w.value[:70] for w in at.warning]}")
        for e in errs:
            print("   ", e)
        failures += len(errs)
    return failures


if __name__ == "__main__":
    load()
    n = run_pages()
    print("FAILURES:", n)
    sys.exit(1 if n else 0)
