import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib, sys

ROOT = Path(__file__).parent.resolve()
CSV = ROOT / "data" / "data.csv"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MODEL = OUT_DIR / "risk_model.pkl"

print(f"[INFO] Loading: {CSV}")
if not CSV.exists():
    print("[ERROR] data/data.csv not found.")
    sys.exit(1)

df = pd.read_csv(CSV, low_memory=False)
print(f"[INFO] Loaded {df.shape[0]} rows, {df.shape[1]} cols")

features = [
    "Age","Income","LoanAmount","CreditScore","MonthsEmployed",
    "InterestRate","LoanTerm","DTIRatio","HasCoSigner"
]
target = "Default"

missing = [c for c in features + [target] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

def to_bool01(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    m = x.map({"yes":1,"y":1,"true":1,"t":1,"1":1,"no":0,"n":0,"false":0,"f":0,"0":0})
    m = m.where(~m.isna(), pd.to_numeric(x, errors="coerce"))
    return m

def to_num(s: pd.Series) -> pd.Series:
    x = (s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip())
    return pd.to_numeric(x, errors="coerce")

clean = pd.DataFrame(index=df.index)
clean["Age"]            = to_num(df["Age"])
clean["Income"]         = to_num(df["Income"])
clean["LoanAmount"]     = to_num(df["LoanAmount"])
clean["CreditScore"]    = to_num(df["CreditScore"])
clean["MonthsEmployed"] = to_num(df["MonthsEmployed"])
clean["InterestRate"]   = to_num(df["InterestRate"])   # e.g., "6.5%" -> 6.5
clean["LoanTerm"]       = to_num(df["LoanTerm"])
clean["DTIRatio"]       = to_num(df["DTIRatio"])       # IMPORTANT: keep as percent (e.g., 35.2)
clean["HasCoSigner"]    = to_bool01(df["HasCoSigner"])

y = df[target]
if y.dtype.kind not in "biu":
    y = to_bool01(y)
y = pd.to_numeric(y, errors="coerce")

keep = ~y.isna()
clean, y = clean.loc[keep].copy(), y.loc[keep].copy()
clean = clean.fillna(clean.median(numeric_only=True))
y = y.astype(int)

print("[INFO] dtypes after cleaning:")
print(clean.dtypes)
print("[INFO] Target distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    clean, y, test_size=0.20, random_state=42, stratify=y
)

clf = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\n[REPORT]")
print(classification_report(y_test, y_pred, digits=4))
try:
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {auc:.4f}")
except Exception:
    pass

joblib.dump(clf, OUT_MODEL)
print(f"[INFO] Saved model → {OUT_MODEL}")
