"""Command-line loader for the Bloomberg workbooks.

Usage (from the project folder, with .venv active):
    python load_bloomberg.py                      # loads every .xlsx in data/input
    python load_bloomberg.py path/to/CP_DATA1.xlsx path/to/CPD.xlsx

Each run stores NEW versions (history is preserved) and makes them current.
"""
import sys
from pathlib import Path

from src import bloomberg as bb
from src.config import INPUT_DIR

paths = [Path(p) for p in sys.argv[1:]] or sorted(INPUT_DIR.glob("*.xls*"))
paths = [p for p in paths if not p.name.startswith("~$")]
if not paths:
    sys.exit(f"No Excel files given and none found in {INPUT_DIR}")
datasets, report = bb.load_workbooks({p.name: p.read_bytes() for p in paths})
for r in report:
    print(f"{r['Sheet'][:28]:28s} → {str(r['Dataset']):18s} {r['Records']:>6} records  {r['Detail']}")
for r in bb.save_all(datasets, ", ".join(p.name for p in paths)):
    print(f"saved {r['dataset']:18s} v{r['version']}  ({r['records']} records)")
print("Done. Start the app with:  python -m streamlit run app.py")
