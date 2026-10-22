import numpy as np
import pandas as pd

FIT_TERM = 36      # fit on 36-month loans only (the 2015 test year is all 36-month)
N_BINS = 10        # quantile bins for numeric features
MIN_LOANS = 500    # bins smaller than this get WoE 0

FEATURES = [       # the 21 features that go into the WoE step
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade", "emp_length",
    "home_ownership", "annual_inc", "purpose", "addr_state", "dti",
    "credit_history_years", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "revol_bal", "revol_util", "total_acc",
    "pub_rec_bankruptcies",
]

# merges decided in notebook 03; any other rare category goes to "rare" automatically
CATEGORY_MERGES = {
    "grade": {"G": "F"},
    "home_ownership": {"OTHER": "RENT", "NONE": "RENT", "ANY": "RENT"},
    "purpose": {
        "car": "A", "credit_card": "A", "major_purchase": "A", "wedding": "A", "home_improvement": "A",
        "debt_consolidation": "B",
        "vacation": "C", "house": "C", "medical": "C", "other": "C",
        "moving": "D", "educational": "D", "renewable_energy": "D",
        "small_business": "E",
    },
}


def fit_population(train):
    # loans the rules are fitted on
    return train[train["term"] == FIT_TERM]


def _fit_rule(col, x, n_bins, min_loans):
    # learn how to bin one feature (train data only)
    if col in CATEGORY_MERGES or not pd.api.types.is_numeric_dtype(x):
        merge = CATEGORY_MERGES.get(col, {})
        merged = x.dropna().map(lambda v: merge.get(v, v))
        counts = merged.value_counts()
        rare = set(counts[counts < min_loans].index)
        groups = {v: ("rare" if merge.get(v, v) in rare else merge.get(v, v)) for v in x.dropna().unique()}
        return {"kind": "category", "groups": groups}

    fill = None
    if x.isnull().sum() < min_loans:          # tiny missing group: treat as typical loans
        fill = x.median()
        x = x.fillna(fill)

    if x.nunique() > n_bins:                  # many values: quantile bins, open-ended at both ends
        _, edges = pd.qcut(x, q=n_bins, retbins=True, duplicates="drop")
        edges[0], edges[-1] = -np.inf, np.inf
        return {"kind": "bins", "edges": edges, "fill": fill}

    counts = x.value_counts()                 # few values: one bin each, rare high values capped
    return {"kind": "values", "cap": counts[counts >= min_loans].index.max(), "fill": fill}


def _assign_bins(x, rule):
    # bin labels (always strings) for any data, using a stored rule
    if rule["kind"] == "category":
        s = x.astype("object")
        grouped = s.map(rule["groups"])                        # unseen value -> NaN
        return grouped.where(s.notna(), "missing").fillna("unseen")

    if rule["fill"] is not None:
        x = x.fillna(rule["fill"])
    if rule["kind"] == "bins":
        codes = pd.cut(x, rule["edges"], labels=False, include_lowest=True)
        labels = codes.astype("Int64").astype(str)             # "0", "1", ...
    else:
        labels = x.clip(upper=rule["cap"]).astype(str)
    return labels.where(x.notna(), "missing")


def _woe_table(labels, y, min_loans):
    # loans, defaults, WoE and IV per bin
    t = y.groupby(labels).agg(loans="count", defaults="sum")
    t["non_defaults"] = t["loans"] - t["defaults"]
    share_good = t["non_defaults"] / t["non_defaults"].sum()
    share_bad = t["defaults"] / t["defaults"].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        t["woe"] = np.log(share_good / share_bad)
    t.loc[t["loans"] < min_loans, "woe"] = 0.0             # too few loans to be evidence
    assert np.isfinite(t["woe"]).all(), "a bin has no defaults or no good loans"
    t["iv"] = (share_good - share_bad) * t["woe"]
    return t


def fit_woe(train, features=FEATURES, n_bins=N_BINS, min_loans=MIN_LOANS):
    # one rule per feature: how to bin it, and the WoE of each bin
    rules = {}
    for col in features:
        rule = _fit_rule(col, train[col], n_bins, min_loans)
        table = _woe_table(_assign_bins(train[col], rule), train["default"], min_loans)
        rule["table"] = table
        rule["woe"] = table["woe"].to_dict()
        rules[col] = rule
    return rules


def apply_woe(df, rules):
    # replace each feature with its bin's WoE; a bin never seen in train gets 0 (average)
    out = pd.DataFrame(index=df.index)
    for col, rule in rules.items():
        out[col] = _assign_bins(df[col], rule).map(rule["woe"]).fillna(0.0).astype(float)
    return out


def iv_summary(rules):
    rows = [{"feature": c, "iv": r["table"]["iv"].sum(), "n_bins": len(r["table"]),
             "thin_bins": int((r["table"]["loans"] < MIN_LOANS).sum())} for c, r in rules.items()]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)