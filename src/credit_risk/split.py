from .data_prep import restrict_vintages, build_resolved, load_needed, clean_types

TEST_YEAR = 2015  # last fully labelled year for 36-month loans; train is every earlier year


def load_model_data(path):
    # load, label, keep usable issue years, convert text columns (notebook 01 steps)
    data = restrict_vintages(build_resolved(load_needed(path)))
    return clean_types(data)


def split_by_year(data, test_year=TEST_YEAR):
    # out-of-time split: train = years before test_year, test = test_year
    # missing values are left as NaN; the WoE step gives them their own bin
    train = data[data["issue_year"] < test_year]
    test = data[data["issue_year"] == test_year]
    return train, test