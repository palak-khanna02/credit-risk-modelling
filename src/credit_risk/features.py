import pandas as pd

NEW_FEATURES = ["loan_to_income", "payment_to_income", "dti_with_loan", "revol_bal_to_income"]
KEPT_FEATURES = ["payment_to_income", "dti_with_loan"] 

def add_features(df):
    # ratio features from credit logic; nothing is fitted, so any loan can be scored
    out = df.copy()
    income = out["annual_inc"].where(out["annual_inc"] > 0)           # zero or missing income -> NaN
    out["loan_to_income"] = out["loan_amnt"] / income
    out["payment_to_income"] = out["installment"] * 12 / income       # yearly payments / yearly income
    out["dti_with_loan"] = out["dti"] + 100 * out["payment_to_income"]  # dti is in percent
    out["revol_bal_to_income"] = out["revol_bal"] / income
    return out