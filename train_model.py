import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.resolve()
CSV = ROOT / "data" / "cleaned_data.csv"  # Changed to cleaned data
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MODEL = OUT_DIR / "risk_model.pkl"
OUT_META = OUT_DIR / "risk_model.meta.json"

FEATURES = [
    "Age","Income","LoanAmount","CreditScore","MonthsEmployed",
    "NumCreditLines","InterestRate","LoanTerm","DTIRatio","HasCoSigner"
]
TARGET = "Default"   # matches your CSV

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

def plot_feature_importance(model, feature_names, out_dir):
    """Plot and save feature importance graph"""
    try:
        import matplotlib.pyplot as plt
        
        importance = model.named_steps['model'].feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance.png")
        plt.close()
        print(f"[INFO] Saved feature importance plot → {out_dir / 'feature_importance.png'}")
    except Exception as e:
        print(f"[INFO] Could not create feature importance plot: {e}")

def load_and_clean(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
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
    # IMPORTANT: DTIRatio is already a ratio (0–1) in your sample; keep as-is.
    clean["DTIRatio"]       = to_num(df["DTIRatio"])
    clean["HasCoSigner"]    = to_bool01(df["HasCoSigner"]).fillna(0).astype(int)

    y = df[TARGET]
    if y.dtype.kind not in "biu":
        y = to_bool01(y)
    y = pd.to_numeric(y, errors="coerce")

    keep = ~y.isna()
    clean, y = clean.loc[keep].copy(), y.loc[keep].copy()
    clean = clean.fillna(clean.median(numeric_only=True))
    y = y.astype(int)

    print("[INFO] dtypes after cleaning:")
    print(clean.dtypes.to_string())
    vc = y.value_counts(dropna=False)
    frac = (vc / vc.sum()).round(4)
    print("[INFO] Target distribution (count | frac):")
    for k in sorted(vc.index):
        print(f"  {k}: {vc[k]} | {frac[k]:.4f}")

    return clean, y

def main():
    X, y = load_and_clean(CSV)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Use RandomForest instead of LogisticRegression
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42
        )),
    ])

    # Add cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC AUC: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n[REPORT]")
    print(classification_report(y_test, y_pred, digits=4))
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        auc = None

    # Add feature importance visualization
    plot_feature_importance(clf, FEATURES, OUT_DIR)

    joblib.dump({"model": clf, "feature_names": FEATURES}, OUT_MODEL)
    print(f"[INFO] Saved model → {OUT_MODEL}")

    meta = {
        "feature_names": FEATURES,
        "target_name": TARGET,
        "notes": "DTIRatio expected as RATIO (0–1).",
        "metrics": {"roc_auc": float(auc) if auc is not None else None},
        "cv_scores": {
            "mean_roc_auc": float(np.mean(cv_scores)),
            "std_roc_auc": float(np.std(cv_scores))
        }
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"[INFO] Saved meta → {OUT_META}")

if __name__ == "__main__":
    main()
