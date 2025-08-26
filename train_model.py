# train_model.py
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

ROOT = Path(__file__).parent.resolve()
CSV = ROOT / "data" / "data_clean.csv"      # use the CLEANED dataset
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MODEL = OUT_DIR / "risk_model.pkl"
OUT_META = OUT_DIR / "risk_model.meta.json"

FEATURES = [
    "Age","Income","LoanAmount","CreditScore","MonthsEmployed",
    "NumCreditLines","InterestRate","LoanTerm","DTIRatio","HasCoSigner"
]
TARGET = "Default"  # must match your CSV

# --------------------- helpers ---------------------
def to_bool01(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    m = x.map({"yes":1,"y":1,"true":1,"t":1,"1":1,"no":0,"n":0,"false":0,"f":0,"0":0})
    m = m.where(~m.isna(), pd.to_numeric(x, errors="coerce"))
    return m

def to_num(s: pd.Series) -> pd.Series:
    x = (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .str.strip()
    )
    return pd.to_numeric(x, errors="coerce")

def plot_feature_importance(model: RandomForestClassifier, feature_names, out_dir: Path):
    """Save feature importance bar chart if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance (RandomForest)")
        plt.bar(range(len(imp)), imp[idx])
        plt.xticks(range(len(imp)), [feature_names[i] for i in idx], rotation=45, ha="right")
        plt.tight_layout()
        out_path = out_dir / "feature_importance.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Saved feature importance plot → {out_path}")
    except Exception as e:
        print(f"[INFO] Skipped feature importance plot: {e}")

def load_and_clean(csv_path: Path):
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found.")
        sys.exit(1)
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[INFO] Loaded {df.shape[0]} rows, {df.shape[1]} cols from {csv_path}")

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    clean = pd.DataFrame(index=df.index)
    clean["Age"]            = to_num(df["Age"])
    clean["Income"]         = to_num(df["Income"])
    clean["LoanAmount"]     = to_num(df["LoanAmount"])
    clean["CreditScore"]    = to_num(df["CreditScore"])
    clean["MonthsEmployed"] = to_num(df["MonthsEmployed"])
    clean["NumCreditLines"] = to_num(df["NumCreditLines"]).fillna(0).astype(int)
    clean["InterestRate"]   = to_num(df["InterestRate"])   # 6.5% -> 6.5
    clean["LoanTerm"]       = to_num(df["LoanTerm"])
    clean["DTIRatio"]       = to_num(df["DTIRatio"])       # expect ratio (0–1)
    clean["HasCoSigner"]    = to_bool01(df["HasCoSigner"]).fillna(0).astype(int)

    # target to 0/1
    y = df[TARGET]
    if y.dtype.kind not in "biu":
        y = to_bool01(y)
    y = pd.to_numeric(y, errors="coerce")

    # ---- DTI unit auto-fix: if median > 1, it was percent; convert to ratio ----
    dti_med = clean["DTIRatio"].median(skipna=True)
    if pd.notna(dti_med) and dti_med > 1:
        print("[WARN] DTIRatio looks like PERCENT; converting to ratio by /100.")
        clean["DTIRatio"] = clean["DTIRatio"] / 100.0
    clean["DTIRatio"] = clean["DTIRatio"].clip(0.0, 1.2)

    # drop rows with missing y, fill remaining numeric NaNs with medians
    keep = ~y.isna()
    clean, y = clean.loc[keep].copy(), y.loc[keep].copy()
    clean = clean.fillna(clean.median(numeric_only=True))
    y = y.astype(int)

    # quick report
    print("[INFO] dtypes after cleaning:")
    print(clean.dtypes.to_string())
    vc = y.value_counts(dropna=False)
    total = int(vc.sum())
    print("[INFO] Target distribution (count | pct):")
    for k in sorted(vc.index):
        pct = vc[k] / total if total else 0
        print(f"  {k}: {vc[k]} | {pct:.4%}")

    return clean, y

# --------------------- main ---------------------
def main():
    X, y = load_and_clean(CSV)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # RandomForest (no scaler needed)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # Cross-validated AUC on train split
    # (Using the estimator directly; CV uses default predict_proba)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"[CV] ROC AUC: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("\n[REPORT]")
    print(classification_report(y_test, y_pred, digits=4))
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        auc = None

    # Optional: save feature importance plot
    plot_feature_importance(rf, FEATURES, OUT_DIR)

    # Save model + feature names (your app expects this shape)
    joblib.dump({"model": rf, "feature_names": FEATURES}, OUT_MODEL)
    print(f"[INFO] Saved model → {OUT_MODEL}")

    meta = {
        "feature_names": FEATURES,
        "target_name": TARGET,
        "notes": "DTIRatio expected as RATIO (0–1); auto-converted if percent detected.",
        "metrics": {"roc_auc": float(auc) if auc is not None else None},
        "cv_scores": {"mean_roc_auc": float(np.mean(cv_scores)), "std_roc_auc": float(np.std(cv_scores))},
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"[INFO] Saved meta → {OUT_META}")

if __name__ == "__main__":
    main()
