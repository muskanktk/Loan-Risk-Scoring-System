# clean_loan_data.py
import pandas as pd
import numpy as np

def clean_loan_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean loan application data with simple business rules.
    Returns a new DataFrame.
    """

    cleaned_df = df.copy()

    # ---- Employment vs Income consistency ----
    unemployed_mask = cleaned_df["EmploymentType"].str.lower().eq("unemployed")
    # Set unemployed income to <= 30k (cap), default to 0 if NaN
    cleaned_df.loc[unemployed_mask, "Income"] = (
        cleaned_df.loc[unemployed_mask, "Income"]
        .fillna(0)
        .apply(lambda x: min(x, 30000))
    )

    # ---- Loan-to-Income ratio sanity ----
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        income_loan_ratio = cleaned_df["LoanAmount"] / cleaned_df["Income"].replace(0, np.nan)
    unrealistic_ratio_mask = income_loan_ratio > 5
    cleaned_df.loc[unrealistic_ratio_mask, "LoanAmount"] = (
        cleaned_df.loc[unrealistic_ratio_mask, "Income"] * 4
    )

    # ---- Clip numeric ranges ----
    cleaned_df["InterestRate"] = cleaned_df["InterestRate"].clip(lower=0, upper=25)
    cleaned_df["CreditScore"] = cleaned_df["CreditScore"].clip(lower=300, upper=850)

    # DTI ratio: keep as ratio (0–1), clip to at most 1.0 (100%)
    cleaned_df["DTIRatio"] = cleaned_df["DTIRatio"].clip(lower=0, upper=1.0)

    # ---- Handle missing values ----
    cleaned_df = cleaned_df.fillna({
        "Income": cleaned_df["Income"].median(),
        "LoanAmount": cleaned_df["LoanAmount"].median(),
        "CreditScore": cleaned_df["CreditScore"].median(),
        "InterestRate": cleaned_df["InterestRate"].median(),
        "DTIRatio": cleaned_df["DTIRatio"].median(),
    })

    return cleaned_df
