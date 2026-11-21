# Asian Paints – AI-Powered Strategic Intelligence Platform

A Streamlit platform for analysing Asian Paints' financial performance, stock behaviour, crude oil and
USD/INR exposure, and brokerage expectations. It also generates evidence-based strategic recommendations
with the Anthropic Claude API.

---

## 1. Setup in Visual Studio Code (Windows / macOS / Linux)

**Prerequisites:** Python 3.10 or newer (3.11 or 3.12 recommended) and VS Code with the *Python* extension.

1. Unzip the project and open the `asian_paints_platform` folder in VS Code (**File → Open Folder**).
2. Open a terminal with **Terminal → New Terminal** and create a virtual environment:

   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   If PowerShell blocks activation, run this first:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your Claude API key. Copy `.env.example` to `.env`, then edit it:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ANTHROPIC_MODEL=claude-sonnet-5-5
   ```
   You can get a key at https://console.anthropic.com. Every module except AI Advisory works without a key.
5. Run the app:
   ```bash
   streamlit run app.py
   ```
   The app opens at http://localhost:8501. You can also press **F5** in VS Code: the launch configuration
   "Streamlit: app.py" is included.
6. In VS Code, choose **Ctrl+Shift+P → Python: Select Interpreter → .venv** so the editor uses the
   environment you just created.

## 2. Your data

Bloomberg workbooks and the local SQLite database are excluded from Git because the workbooks may be subject to
Bloomberg licensing terms and the database is generated locally. To use the automatic first-run import, place your
authorized workbooks in **`data/input/`**:

| Workbook | Sheet(s) | Loaded as |
|---|---|---|
| `CP_DATA1.xlsx` | USDINR (INR REGN Curncy) | USD/INR daily |
| | Brent Crude EU (EUCRBRDT Index) | Brent spot daily |
| | Brent Crude US (**COZ6 Comdty**) | Brent *futures* Dec-26 contract. **This is not WTI**, so it's stored separately. |
| | Asian Paints Daily OHLC (APNT IN) | Daily open, high, low and close (the file has no volume column) |
| | IS / BS / CFS All time | Annual statements FY2002–FY2026, converted from INR million to ₹ Crore |
| `CPD.xlsx` | 42 × "Company Financial (Single Period)" | Consensus vs actuals for FY18 Q1 → FY26 Q2 (quarterly and yearly), the analyst low–high range, and a **quarterly P&L** that includes raw-material cost |

If authorized workbooks are present in `data/input/`, **the app loads them automatically** when the database is empty.
To load them later, or to load updated exports, use one of these:
- Go to **Data Management Centre → Bloomberg import**, click **Analyse workbooks**, then **Save all**. You can also upload new files there.
- Or run `python load_bloomberg.py` from the terminal (or `python load_bloomberg.py path\to\file.xlsx`).

Every reload creates new versions. Old versions stay in **Versions & history**.

Notes on the data:
- **Units:** INR million is converted to ₹ Crore (× 0.1). EPS, DPS, percentages, ratios and share counts are not converted.
- **Raw-material cost** is missing from Bloomberg's standardized annual IS, so it's taken from the consensus
  workbook's actuals (FY2018–FY2025 annually, plus every quarter). Gross margin therefore starts in FY2018.
- **Estimate columns** (`FY 2027 Est`, `FY 2028 Est`) are kept apart as "street consensus" and drawn on forecast
  charts for comparison. They're never mixed with actuals. The `Last 12M` column is skipped.
- **Data-quality check:** in four quarters the vendor's reported gross margin disagrees with revenue minus material
  cost. One quarter, Sep-2018, is also an outlier against the quarters around it, so it's excluded from the
  crude-to-margin regression by default. A switch on that page lets you include it.
- **Fiscal quarters:** Q1 = Apr–Jun. Bloomberg's "2026:Q2" means the quarter ending Sep-2025.

## 2b. Adding other files later
Use **Upload & map (any file)** for non-Bloomberg CSV or Excel files, such as a WTI series or peer financials.

## 3. Modules

| Page | What it does |
|---|---|
| Executive Dashboard | KPIs for share price, fundamentals, crude and FX, and beat/miss against estimates; the link between crude and margin |
| Macroeconomic Intelligence | Trends, Brent in ₹, volatility, correlation matrix, rolling and lagged correlation, multiple regression with HAC errors and VIF |
| Stock Analysis | Candlesticks, moving averages, Bollinger bands, RSI, MACD, risk statistics, drawdown, and peer benchmarking (Berger, Kansai, Indigo, Akzo, Nifty) |
| Financial Statements | Reported values, YoY/QoQ growth, common-size view, CAGR |
| Financial Ratios | Margins, ROE, ROCE, leverage, liquidity, working-capital days, DuPont analysis, cash quality |
| Expectations vs Actuals | Surprise %, beat/miss/in-line counts, streaks, heatmap, share-price reaction |
| Predictive & Scenarios | ETS/ARIMA/trend forecasts with walk-forward backtests against a naive forecast; ML return model; crude/FX cost scenarios; sensitivity and tornado charts; DCF |
| AI Strategic Advisory | Claude produces 8-part recommendations from a Python-calculated evidence pack, with each claim labelled FACT, FORECAST, SCENARIO or OPINION |
| Research Reports | HTML report (print to PDF from the browser), Markdown, and an Excel data pack |
| Data Management Centre | Upload, mapping, validation, versions, manual edits, API fetch, export, audit log |

## 4. Project structure

```
asian_paints_platform/
├── app.py                  # entry point + navigation
├── requirements.txt
├── .env.example            # copy to .env
├── .streamlit/config.toml  # theme (dark default; switch in ⋮ → Settings)
├── .vscode/launch.json     # F5 to run
├── src/
│   ├── config.py           # paths, dataset catalogue, units, tickers
│   ├── database.py         # SQLite, versioned datasets, audit log
│   ├── ingestion.py        # file reading, header/date/unit detection, mapping
│   ├── validation.py       # data-quality checks
│   ├── market_data.py      # yfinance fetchers
│   ├── data_access.py      # current-data helpers, line-item resolution
│   ├── analytics.py        # technicals, growth, correlation, regression, ratios
│   ├── forecasting.py      # time-series + ML with leakage-safe validation
│   ├── scenario.py         # cost scenarios, sensitivity, DCF
│   ├── insights.py         # evidence pack for AI and reports
│   ├── ai_advisor.py       # Claude integration
│   ├── reports.py          # HTML / Markdown / Excel
│   └── ui.py               # theme, KPI cards, chart styling
├── views/                  # one file per page
├── data/db/platform.sqlite # created on first run
├── data/raw/               # copies of every uploaded file
└── tests/                  # synthetic-data generator + end-to-end test
```

## 5. Testing

```bash
python tests/test_pipeline.py        # loads data/input into a temporary DB and renders every page
python tests/make_sample_data.py     # optional: synthetic messy files for the generic uploader
python tests/test_generic_upload.py
```

## 6. Methodology safeguards

- **No look-ahead bias:** lagged analysis uses only past values of each driver. Forecasts are checked with
  expanding-window walk-forward tests. The ML model uses `TimeSeriesSplit`, fits its scaler inside each fold
  and measures its baseline on training data only.
- **No spurious correlation:** inference uses log returns. The page warns when you switch to price levels.
- **Honest benchmarks:** every forecast is compared with a naive forecast, and the page says plainly when
  the model fails to beat it.
- **AI grounding:** Claude receives only calculated numbers and must label each one. It does not recompute
  financials.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate `.venv`, then run `pip install -r requirements.txt` |
| Fetching from Yahoo returns nothing | Check your internet connection. Yahoo sometimes rate-limits requests, so try again later. |
| A ratio shows "–" | Correct the line item in Data Management → Line-item mapping |
| Dates look wrong | Check the "Date format ambiguity" warning during upload; DD/MM is assumed when the format is ambiguous |
| A Claude 401 error | The key in `.env` is wrong. Restart Streamlit after editing `.env`. |
| Wrong model name | Set `ANTHROPIC_MODEL` in `.env` or in the AI page sidebar |

## 8. Deployment (optional)

To deploy on Streamlit Community Cloud, push the project to GitHub (without `.env`), create an app that
points to `app.py`, and add `ANTHROPIC_API_KEY` under **Secrets**. SQLite storage there is temporary; for
lasting storage, set `APP_DB_PATH` to a mounted volume or move to Postgres.
