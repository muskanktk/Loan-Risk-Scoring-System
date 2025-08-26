# clean_loan_data.py
import pandas as pd
import numpy as np

NUMERIC_BOUNDS = {
    "Income": (0, 1_000_000),
    "LoanAmount": (0, 2_000_000),
    "CreditScore": (300, 850),
    "InterestRate": (0, 25),     # keep generous upper bound
    "LoanTerm": (6, 600),        # months: 6–50 years (generous)
    "DTIRatio": (0.0, 1.2),      # treat as ratio; clip > 120% to 1.2 then normalize to ≤1.0 later
    "MonthsEmployed": (0, 600),  # up to 50 years
    "NumCreditLines": (0, 60),
}

YES_NO_COLS = ["HasMortgage", "HasDependents", "HasCoSigner"]
CATEGORICAL_COLS = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]

def _to_num(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .str.strip()
         .replace({"": np.nan, "nan": np.nan, "None": np.nan})
         .pipe(pd.to_numeric, errors="coerce")
    )

def _to_bool01(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    m = x.map({"yes":1,"y":1,"true":1,"t":1,"1":1,"no":0,"n":0,"false":0,"f":0,"0":0})
    # Try numeric fallback
    m = m.where(~m.isna(), pd.to_numeric(x, errors="coerce"))
    return m

def _clip(df: pd.DataFrame, col: str, lo: float, hi: float):
    if col in df.columns:
        df[col] = df[col].clip(lower=lo, upper=hi)

def clean_loan_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean loan application data with sanity rules and unit normalization.
    Returns a NEW DataFrame ready for model training (numeric/binary kept as-is).
    """
    d = df.copy()

    # --- Coerce core numerics ---
    for col in ["Age","Income","LoanAmount","CreditScore","MonthsEmployed",
                "NumCreditLines","InterestRate","LoanTerm","DTIRatio"]:
        if col in d.columns:
            d[col] = _to_num(d[col])

    # --- Coerce yes/no flags to 0/1 (keep NaN for now) ---
    for col in YES_NO_COLS:
        if col in d.columns:
            d[col] = _to_bool01(d[col])

    # --- EmploymentType rule: unemployed incomes shouldn’t be huge ---
    if "EmploymentType" in d.columns and "Income" in d.columns:
        unemployed_mask = d["EmploymentType"].astype(str).str.lower().eq("unemployed")
        d.loc[unemployed_mask, "Income"] = (
            d.loc[unemployed_mask, "Income"].fillna(0).clip(0, 30_000)
        )

    # --- LTI sanity: if LoanAmount/Income > 5, reduce to 4×Income (avoid div-by-zero) ---
    if "LoanAmount" in d.columns and "Income" in d.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            lti = d["LoanAmount"] / d["Income"].replace(0, np.nan)
        too_high = lti > 5
        d.loc[too_high, "LoanAmount"] = d.loc[too_high, "Income"] * 4

    # --- Clip numeric ranges (broad, safe) ---
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col in d.columns:
            _clip(d, col, lo, hi)

    # --- Normalize DTI units: detect percent vs ratio ---
    # If median > 1, likely percent → convert to ratio
    if "DTIRatio" in d.columns:
        med = d["DTIRatio"].median(skipna=True)
        if pd.notna(med) and med > 1:
            # treat values like 35 (35%) as 0.35
            d["DTIRatio"] = d["DTIRatio"] / 100.0
        # Final clamp to [0, 1.0]
        d["DTIRatio"] = d["DTIRatio"].clip(0.0, 1.0)

    # --- Basic impossible-row filter (optional, conservative) ---
    # drop rows that still have completely unusable values
    drop_mask = pd.Series(False, index=d.index)
    for col in ["Income","LoanAmount","CreditScore","InterestRate","LoanTerm","DTIRatio"]:
        if col in d.columns:
            drop_mask |= d[col].isna()
    d = d.loc[~drop_mask].copy()

    # --- Fill remaining NaNs in yes/no flags with 0 (conservative) ---
    for col in YES_NO_COLS:
        if col in d.columns:
            d[col] = d[col].fillna(0).astype(int)

    # --- Ensure integer types where appropriate ---
    for col in ["MonthsEmployed","NumCreditLines","LoanTerm"]:
        if col in d.columns:
            d[col] = d[col].round().astype(int)

    # CreditScore already clipped to [300,850]; round to integer
    if "CreditScore" in d.columns:
        d["CreditScore"] = d["CreditScore"].round().astype(int)

    return d
