# make_clean_dataset.py
from pathlib import Path
import pandas as pd
from clean_loan_data import clean_loan_data

ROOT = Path(__file__).parent.resolve()
RAW = ROOT / "data" / "data.csv"           # your current raw file
CLEAN = ROOT / "data" / "data_clean.csv"   # new cleaned file (you can choose to overwrite data.csv instead)

if not RAW.exists():
    raise SystemExit(f"[ERROR] {RAW} does not exist")

print(f"[INFO] Reading raw: {RAW}")
df = pd.read_csv(RAW, low_memory=False)

print("[INFO] Cleaning...")
clean = clean_loan_data(df)

print(f"[INFO] Writing cleaned → {CLEAN}  (rows: {len(clean)})")
clean.to_csv(CLEAN, index=False)

print("[DONE] You can now train on data/data_clean.csv")
