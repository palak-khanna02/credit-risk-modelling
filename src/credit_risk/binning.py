import json
from pathlib import Path
import numpy as np
from . import woe as wo
from . import features as ft

# cut points saved from notebook 05 (optbinning, at most 6 bins)
BINS = json.loads((Path(__file__).parent / "bins.json").read_text())


def numeric_rule(x, cuts):
    # bin rule from stored cut points; a missing group under 500 loans is filled with the median
    fill = x.median() if x.isnull().sum() < wo.MIN_LOANS else None
    edges = np.array([-np.inf] + list(cuts) + [np.inf])
    return {"kind": "bins", "edges": edges, "fill": fill}

def fit_final(train):
    # one rule per feature: stored cut points for numeric, the woe.py groups for categories
    rules = {}
    for col in wo.FEATURES + ft.KEPT_FEATURES:
        if col in BINS:
            rule = numeric_rule(train[col], BINS[col])
        else:
            rule = wo._fit_rule(col, train[col], wo.N_BINS, wo.MIN_LOANS)
        table = wo._woe_table(wo._assign_bins(train[col], rule), train["default"], wo.MIN_LOANS)
        rule["table"], rule["woe"] = table, table["woe"].to_dict()
        rules[col] = rule
    return rules