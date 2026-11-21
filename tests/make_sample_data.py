"""Generate SYNTHETIC test files in deliberately messy, varied formats.

These are NOT real Asian Paints data. They exist only to test the ingestion layer
(title rows, Indian date formats, FRED '.' missing values, periods-as-columns, FY/Q labels).
Run:  python tests/make_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "sample_data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
days = pd.bdate_range("2012-01-02", "2025-10-31")


def walk(start, vol, n, drift=0.0002):
    return start * np.exp(np.cumsum(rng.normal(drift, vol, n)))


brent = walk(110, 0.022, len(days), 0)
wti = brent - 4 + rng.normal(0, 1, len(days))
fx = walk(50, 0.004, len(days), 0.00018)

# USD/INR – FRED style with "." as missing
f = pd.DataFrame({"DATE": days.strftime("%Y-%m-%d"), "DEXINUS": np.round(fx, 4).astype(str)})
f.loc[f.sample(30, random_state=1).index, "DEXINUS"] = "."
f.to_csv(OUT / "usd_inr.csv", index=False)

# Brent – Excel with title rows, dd-mm-yyyy text dates
with pd.ExcelWriter(OUT / "brent_europe.xlsx") as xw:
    pd.DataFrame([["Europe Brent Spot Price FOB (Dollars per Barrel)"], ["Source: synthetic"]]).to_excel(
        xw, index=False, header=False)
    pd.DataFrame({"Date": days.strftime("%d-%m-%Y"), "Price ($/bbl)": np.round(brent, 2)}).to_excel(
        xw, index=False, startrow=3)

# WTI – CSV mm/dd/yyyy
pd.DataFrame({"Date": days.strftime("%m/%d/%Y"), "Cushing OK WTI Spot Price": np.round(wti, 2)}).to_csv(
    OUT / "wti_us.csv", index=False)

# OHLC – NSE style column names with trailing spaces, 01-Jan-2020
close = walk(400, 0.015, len(days), 0.0005)
op = close * (1 + rng.normal(0, 0.005, len(days)))
hi = np.maximum(op, close) * (1 + abs(rng.normal(0, 0.006, len(days))))
lo = np.minimum(op, close) * (1 - abs(rng.normal(0, 0.006, len(days))))
pd.DataFrame({"Date ": days.strftime("%d-%b-%Y"), "series ": "EQ", "OPEN ": op.round(2), "HIGH ": hi.round(2),
              "LOW ": lo.round(2), "PREV. CLOSE ": np.r_[np.nan, close[:-1]].round(2), "ltp ": close.round(2),
              "close ": close.round(2), "VOLUME ": rng.integers(5e5, 3e6, len(days)).astype(str)}
             ).to_csv(OUT / "asian_paints_ohlc.csv", index=False)

# Annual statements – items in rows, "Mar 2013".."Mar 2025", Rs. Cr in title
years = list(range(2013, 2026))
cols = [f"Mar {y}" for y in years]
rev = np.array([12000 * 1.11 ** i for i in range(len(years))]) * (1 + rng.normal(0, 0.02, len(years)))
bavg = pd.Series(brent, index=days).groupby(lambda d: d.year + (1 if d.month > 3 else 0)).mean()
fxavg = pd.Series(fx, index=days).groupby(lambda d: d.year + (1 if d.month > 3 else 0)).mean()
crude_inr = np.array([bavg.get(y, bavg.iloc[-1]) * fxavg.get(y, fxavg.iloc[-1]) for y in years])
mat_pct = 0.50 + 0.12 * (crude_inr / crude_inr.mean() - 1) + rng.normal(0, 0.01, len(years))
rm = rev * mat_pct * 0.85
pur = rev * mat_pct * 0.12
invc = rev * mat_pct * 0.03
emp = rev * 0.065
oth = rev * 0.17
oi = rev * 0.02
dep = rev * 0.03
fin = rev * 0.008
pbt = rev - rm - pur - invc - emp - oth + oi - dep - fin
tax = pbt * 0.255
pat = pbt - tax
inc = pd.DataFrame({
    "Revenue From Operations": rev, "Other Income": oi, "Total Income": rev + oi,
    "Cost Of Materials Consumed": rm, "Purchase Of Stock-In Trade": pur,
    "Changes In Inventories Of FG,WIP And Stock-In Trade": invc, "Employee Benefit Expenses": emp,
    "Finance Costs": fin, "Depreciation And Amortisation Expenses": dep, "Other Expenses": oth,
    "Profit/Loss Before Tax": pbt, "Total Tax Expenses": tax, "Profit/Loss For The Period": pat,
    "Basic EPS (Rs.)": pat / 95.92}, index=cols).T.round(2)
with pd.ExcelWriter(OUT / "income_statement.xlsx") as xw:
    pd.DataFrame([["Asian Paints – Consolidated Profit & Loss (Rs. Cr)"]]).to_excel(xw, index=False, header=False)
    inc.reset_index(names="Particulars").to_excel(xw, index=False, startrow=2)

eq = np.cumsum(pat * 0.45) + 6000
debt = rev * 0.04
ta = eq + debt + rev * 0.25
bs = pd.DataFrame({"Equity Share Capital": np.full(len(years), 95.92), "Total Equity": eq,
                   "Long Term Borrowings": debt * 0.4, "Short Term Borrowings": debt * 0.6,
                   "Trade Payables": rev * 0.12, "Total Current Liabilities": rev * 0.22,
                   "Inventories": rev * 0.16, "Trade Receivables": rev * 0.09,
                   "Cash And Cash Equivalents": rev * 0.03, "Current Investments": rev * 0.08,
                   "Total Current Assets": rev * 0.42, "Total Assets": ta,
                   "Total Equity And Liabilities": ta}, index=cols).T.round(2)
bs.reset_index(names="Balance Sheet (Rs. Cr)").to_csv(OUT / "balance_sheet.csv", index=False)

capex = -rev * 0.05
cf = pd.DataFrame({"Net Cash From Operating Activities": pat * 1.05, "Purchase Of Property, Plant And Equipment": capex,
                   "Net Cash Used In Investing Activities": capex * 1.3, "Dividends Paid": -pat * 0.5,
                   "Net Cash Used In Financing Activities": -pat * 0.6}, index=[f"FY{str(y)[2:]}" for y in years]).T.round(2)
cf.reset_index(names="Cash Flow (Rs. in Crores)").to_csv(OUT / "cash_flow.csv", index=False)

rat = pd.DataFrame({"Year": [f"{y-1}-{str(y)[2:]}" for y in years], "ROE (%)": (pat / eq * 100).round(2),
                    "Net Profit Margin (%)": (pat / rev * 100).round(2), "Current Ratio (x)": np.round(0.42 / 0.22, 2),
                    "Inventory Turnover (x)": (rev / (rev * 0.16)).round(2)})
rat.to_excel(OUT / "financial_ratios.xlsx", index=False)

# Quarterly brokerage – wide paired columns, Q1FY22 style
qs = [f"Q{q}FY{y}" for y in range(20, 26) for q in (1, 2, 3, 4)]
exp_rev = np.array([5000 * 1.025 ** i for i in range(len(qs))])
act_rev = exp_rev * (1 + rng.normal(0.005, 0.03, len(qs)))
exp_e = exp_rev * 0.2
act_e = exp_e * (1 + rng.normal(-0.01, 0.06, len(qs)))
exp_p = exp_e * 0.65
act_p = exp_p * (1 + rng.normal(0, 0.07, len(qs)))
pd.DataFrame({"Quarter": qs, "Revenue Expected": exp_rev.round(1), "Revenue Actual": act_rev.round(1),
              "EBITDA Estimate": exp_e.round(1), "EBITDA Actual": act_e.round(1),
              "PAT Consensus": exp_p.round(1), "PAT Actual": act_p.round(1),
              "Remarks": ["Beat" if a > e else "Miss" for a, e in zip(act_p, exp_p)]}
             ).to_excel(OUT / "brokerage_quarterly.xlsx", index=False)

# Yearly brokerage – long format
rows = []
for y in range(2018, 2026):
    for m, base in [("Revenue", 20000), ("EBITDA", 4000), ("PAT", 2800)]:
        e = base * 1.1 ** (y - 2018)
        rows.append({"Fiscal Year": f"FY{y}", "Parameter": m, "Street Expectation": round(e, 1),
                     "Actual Reported": round(e * (1 + rng.normal(0, 0.04)), 1)})
pd.DataFrame(rows).to_csv(OUT / "brokerage_yearly.csv", index=False)
print("Synthetic files written to", OUT)
