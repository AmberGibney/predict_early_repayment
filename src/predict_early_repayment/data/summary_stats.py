from typing import List

import pandas as pd
from scipy.stats import chi2_contingency


def check_and_return_spearman_corr(df: pd.DataFrame, thresh: float = 0.4):
    """Function that checks for spearman correlations (continuous and categorical rank
    association strength) and returns anything over a chosen threshold (defalt of
    moderate > 0.4)"""

    # Training set high correlations - ob

    df_corr = (
        df[df["which"] == 1]
        .corr(numeric_only=True)
        .abs()
        .unstack()
        .sort_values(kind="quicksort", ascending=False)
        .reset_index()
    )
    df_corr.rename(
        columns={
            "level_0": "Feature 1",
            "level_1": "Feature 2",
            0: "Correlation Coefficient",
        },
        inplace=True,
    )
    df_corr.drop(df_corr.iloc[1::2].index, inplace=True)
    df_corr_nd = df_corr.drop(df_corr[df_corr["Correlation Coefficient"] == 1.0].index)

    # set threshold at 0.4 for a moderate correlation
    corr = df_corr_nd["Correlation Coefficient"] > thresh

    return df_corr_nd[corr]


def check_categorical_variables(df: pd.DataFrame, cols: List):
    """Function to check chi-square contingency tables for catgeorical columns against
    the target."""

    for i in cols:
        ct = pd.crosstab(columns=df[df["which"] == 1][i], index=df["early_settled"])
        stat, p, dof, expected = chi2_contingency(ct)

        # print contingency tables
        if p < 0.05:
            print(ct)
            print("Potential association with target for {}".format(i))

        else:
            print(ct)


def compare_continous_by_target(df: pd.DataFrame):
    """ "Function to compare values of continuous variables for each outcome group."""

    var_cont = [
        x
        for x in df.columns
        if x not in ["loan_id", "origination_date", "early_settled"]
    ]

    return (
        df[df["which"] == 1][var_cont + ["early_settled"]]
        .groupby(["early_settled"])
        .mean(numeric_only=True)
    )
