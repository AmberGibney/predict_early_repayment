import logging
from datetime import timedelta
from functools import reduce
from typing import List

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from predict_early_repayment.utils import load_from_json

logger = logging.getLogger(__name__)


def filter_customer_by_time(df: pd.DataFrame, years_num: int):
    """ "Function to filter tradeline dataset for years per customer."""

    logging.info(f"Filtering tradeline data for {years_num} years per customer")

    df_filtered = (
        df.groupby("loan_id")
        .apply(
            lambda g: g[g["date"] > (g["date"].max() - relativedelta(years=years_num))]
        )
        .reset_index(level=0, drop=True)
    )

    return df_filtered


def get_latest_product_balance(
    df: pd.DataFrame, customer: pd.DataFrame, product: str, num_recent_days: int
):
    # filter for account type, sort and get most recent dated row per id
    df = (
        df[df["account_type"] == product]
        .sort_values(by="date")
        .groupby("loan_id")
        .tail(1)
    )

    # join with customer dataset
    df = df.merge(customer[["loan_id", "origination_date"]], how="left")

    # keep rows that are within num_recent_months of loan origination date
    df = df[abs(df["origination_date"] - df["date"]) < timedelta(days=num_recent_days)]

    # flag for whether current debt is larger than opening balance
    if product in ["Mortgage", "Loan"]:
        df["debt_increasing"] = np.where(
            (df["opening_balance"] != 0.0) & (df["balance"] > df["opening_balance"]),
            1,
            0,
        )

        # df["prop_remains"] = df["balance"]/df["opening_balance"]

    # return cols we want
    return df[[x for x in ["loan_id", "balance", "debt_increasing"] if x in df.columns]]


def merge_latest_product_balance(
    df: pd.DataFrame, customer: pd.DataFrame, product: str
):
    # set df col names to represent product
    base_cols = ["loan_id", f"{product.lower()}_balance"]

    df.columns = (
        base_cols + [f"{product.lower()}_debt_increasing"]
        if product in ["Mortgage", "Loan"]
        else base_cols
    )

    return customer.merge(df, how="left")


def add_latest_balance(
    df: pd.DataFrame, customer: pd.DataFrame, product: str, num_recent_days: int = 90
):
    """ "Wrapper function to get latest product balance and add it to the customer
    table."""

    latest = get_latest_product_balance(df, customer, product, num_recent_days)

    logging.info(f"Adding latest balance for {product}")

    return merge_latest_product_balance(latest, customer, product)


def get_max_status(df: pd.DataFrame, new_col: str):
    """ ""Function to get customer's max tradeline status as a new named column."""

    logging.info(f"Calculating column {new_col}")

    return df.groupby("loan_id")["status"].max().rename(new_col).reset_index()


def get_overdraft_usage(df: pd.DataFrame):
    """Function to calculate the number of months that a customer has used their
    overdraft in the dataset."""

    logging.info("Calculating number of months overdraft usage")
    return (
        df[(df["account_type"] == "CurrentAccount") & (df["balance"] != 0.0)]
        .groupby("loan_id")["date"]
        .nunique()
        .rename("months_overdraft")
        .reset_index()
    )


def calculate_total_debt(df: pd.DataFrame):
    # get total current debt
    logging.info("Calculating current total debt")

    # create balance cols to filter
    balance_cols = [x for x in df.columns if "balance" in x]

    # calculate pre-existing debt
    df["prexisting_debt"] = df[balance_cols].sum(axis=1)

    # calculate total debt including new loan
    # df["total_debt"] = df["prexisting_debt"]  + df["amount"]

    # calculate debt to income ratio if we can find term
    # df["debt_to_inc"] = (df["total_debt"]/total_term)/df["income"]

    return df.drop(columns=balance_cols)


def any_increasing_debt(df: pd.DataFrame):
    """Function to create single column to indicate if existing mortgage or loan balance
    is larger than opening debt."""

    # create new column
    df["existing_increasing_debt"] = np.where(
        (df["mortgage_debt_increasing"] == 1) | (df["loan_debt_increasing"] == 1), 1, 0
    )

    return df.drop(columns=["mortgage_debt_increasing", "loan_debt_increasing"])


def create_custom_features(tradeline: pd.DataFrame, cust: pd.DataFrame):
    """Wrapper function to create new features."""

    # add latest balance from accounts
    for account in ["Mortgage", "Loan", "CurrentAccount", "CreditCard"]:
        cust = add_latest_balance(tradeline, cust, account)

    # add flag for increasing debt

    # get total current debt
    cust = calculate_total_debt(cust)

    # calculate increasing debt
    cust = any_increasing_debt(cust)

    # create filtered version of tradelines for 2 years per customer
    trade_year = filter_customer_by_time(tradeline, 2)

    # create other features
    max_2_year_status = get_max_status(trade_year, "max_status_2y")
    max_status = get_max_status(trade_year, "max_status_alltime")
    overdraft_2y = get_overdraft_usage(trade_year)

    return [cust, max_2_year_status, max_status, overdraft_2y]


def add_custom_features_to_dataset(dfs: List):
    """Function to merge custom feature datasets into customer dataset."""

    logging.info("Adding custom features to dataset")

    df = reduce(
        lambda left, right: pd.merge(left, right, on="loan_id", how="left"), dfs
    )

    # get dummies
    df = pd.get_dummies(
        data=df,
        columns=["home_ownership", "employment_status"],
        dtype=float,
        drop_first=True,
    )

    # process column names
    df.columns = df.columns.str.replace(" ", "")

    # load na_cols
    na_cols = load_from_json("commons/na_cols.json")["na_cols"]

    return _fillna_in_selected_cols(df, na_cols)


def _fillna_in_selected_cols(df: pd.DataFrame, na_cols: List):
    """Function to fill na into selected columns."""

    logging.info(f"Filling 0 into NA in {na_cols}")

    df[na_cols] = df[na_cols].fillna(0)

    return df
