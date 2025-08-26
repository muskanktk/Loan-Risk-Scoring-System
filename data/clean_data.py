# clean_loan_data.py
import pandas as pd
import numpy as np

def clean_loan_data(df):
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Fix employment-income inconsistencies
    unemployed_mask = cleaned_df['EmploymentType'] == 'Unemployed'
    cleaned_df.loc[unemployed_mask, 'Income'] = cleaned_df.loc[unemployed_mask, 'Income'].apply(
        lambda x: min(x, 30000) 
    )
    
    income_loan_ratio = cleaned_df['LoanAmount'] / cleaned_df['Income']
    unrealistic_ratio_mask = income_loan_ratio > 5
    cleaned_df.loc[unrealistic_ratio_mask, 'LoanAmount'] = cleaned_df.loc[unrealistic_ratio_mask, 'Income'] * 4
    
    cleaned_df['InterestRate'] = cleaned_df['InterestRate'].clip(upper=15)
    
    cleaned_df['CreditScore'] = cleaned_df['CreditScore'].clip(lower=300, upper=850)
    
    cleaned_df['DTIRatio'] = cleaned_df['DTIRatio'].clip(upper=0.8)  # Cap at 80%
    
    return cleaned_df
