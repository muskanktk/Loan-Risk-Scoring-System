from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
CLEAN = ROOT / "data" / "data_clean.csv"
TAB = ROOT / "data" / "loans_tableau.csv"

if not CLEAN.exists():
    raise SystemExit(f"[ERROR] {CLEAN} not found. Run make_clean_dataset.py first.")

df = pd.read_csv(CLEAN, low_memory=False)

keep = [
    "Age","Income","LoanAmount","CreditScore","MonthsEmployed","NumCreditLines",
    "InterestRate","LoanTerm","DTIRatio","HasCoSigner",
    # add any business fields you have (state, grade, purpose, issue_date, Default, loss, etc.)
    "Default",
]
df = df[[c for c in keep if c in df.columns]].copy()

df["EmploymentYears"] = (df["MonthsEmployed"] / 12).round(1)
df.to_csv(TAB, index=False)
print(f"[OK] Wrote {TAB}  (rows={len(df)})")
