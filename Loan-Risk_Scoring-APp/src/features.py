import pandas as pd
from .config import DTI_INPUT_IS_RATIO

def build_row(age, income, loan_amount, credit_score, months_employed,
              interest_rate, term_months, dti_ratio, has_cosigner):
    dti_for_model = dti_ratio*100.0 if DTI_INPUT_IS_RATIO else dti_ratio
    return pd.DataFrame([{
        "Age": age,
        "Income": float(income),
        "LoanAmount": float(loan_amount),
        "CreditScore": int(credit_score),
        "MonthsEmployed": int(months_employed),
        "InterestRate": float(interest_rate),
        "LoanTerm": int(term_months),
        "DTIRatio": float(dti_for_model),
        "HasCoSigner": int(has_cosigner),
    }])
