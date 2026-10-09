import pandas as pd 

FINAL_STATUSES = [ 
    "Fully Paid",
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
]

DEFAULT_STATUSES = [  
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
]

ID_COLS = ['id' ]  

FEATURE_COLS = [  #  one flat list, the 31 features from notebook 01
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status", "purpose",
    "addr_state", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "collections_12_mths_ex_med", "application_type",
    "acc_now_delinq", "chargeoff_within_12_mths", "delinq_amnt", "pub_rec_bankruptcies", "tax_liens",
]

# new: column names after clean_types (earliest_cr_line becomes credit_history_years)
MODEL_FEATURES = ["credit_history_years" if c == "earliest_cr_line" else c for c in FEATURE_COLS]

TARGET_COLS = ["loan_status", "issue_d"]

LGD_COLS = ["funded_amnt", "recoveries", "collection_recovery_fee", "total_rec_prncp", "out_prncp"]  


MIN_YEAR = 2007                          # earliest usable vintage
LAST_CLEAN_YEAR = {36: 2015, 60: 2013}   # last fully resolved vintage, per term (months)

EMP_LENGTH_MAP = {"< 1 year": 0, "1 year": 1, "10+ years": 10,  # new
                  **{f"{i} years": i for i in range(2, 10)}}

def load_raw(path, nrows=None, usecols=None):
    return pd.read_csv(path, low_memory=False, nrows=nrows, usecols=usecols)

def load_needed(path, nrows=None):  # loads only the columns we use
    cols = ID_COLS + TARGET_COLS + LGD_COLS + FEATURE_COLS
    df = load_raw(path, nrows=nrows, usecols=cols)
    df["term"] = df["term"].str.strip()  # new: harmless if there is no leading space
    return df

def build_resolved(df):
    # keeps loans with a known outcome (this also drops the 33 non-loan rows), adds 0/1 'default'
    resolved = df[df["loan_status"].isin(FINAL_STATUSES)].copy()
    resolved["default"] = resolved["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    return resolved

def restrict_vintages(resolved):
    out = resolved.copy()
    out["issue_year"] = pd.to_datetime(out["issue_d"], format="%b-%Y").dt.year
    term_months = out["term"].str.extract(r"(\d+)")[0].astype(int)
    last_year = term_months.map(LAST_CLEAN_YEAR)
    keep = (out["issue_year"] >= MIN_YEAR) & (out["issue_year"] <= last_year)
    return out[keep]

def clean_types(df):  # text columns to numbers, no imputation
    out = df.copy()
    out["term"] = out["term"].str.extract(r"(\d+)")[0].astype(int)   # "36 months" -> 36
    out["emp_length"] = out["emp_length"].map(EMP_LENGTH_MAP)        # missing stays NaN
    issue = pd.to_datetime(out["issue_d"], format="%b-%Y")
    earliest = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y")
    out["credit_history_years"] = (issue - earliest).dt.days / 365.25
    return out.drop(columns=["earliest_cr_line"])

def get_X_y(df):  #  expects a dataframe that has been through clean_types
    return df[MODEL_FEATURES], df["default"]

