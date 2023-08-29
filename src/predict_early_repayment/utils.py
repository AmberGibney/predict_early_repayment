import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pkg_resources
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_from_json(path_name: str):
    """Function to load list from json in commons folder."""

    # set path
    path = pkg_resources.resource_string("predict_early_repayment", path_name)

    # load json
    return json.loads(path.decode())


def check_loans_data_post_tradeline(customer: pd.DataFrame, tradeline: pd.DataFrame):
    """Function to check that all loans in customer table originate after loans
    tradeline history ie.

    that they don't describe the same loans

    """

    # filter tradeline for loans
    df = tradeline[(tradeline["account_type"] == "Loan")][
        ["loan_id", "date", "opening_balance"]
    ].drop_duplicates()

    # merge with customer data
    df = df.merge(
        customer[["loan_id", "origination_date"]],
        left_on=["loan_id"],
        right_on=["loan_id"],
    )

    # filter for tradeline date after loan date
    df = df[df["date"] >= df["origination_date"]]

    if len(df) == 0:
        logging.info("Tradeline data pre-customer data - okay")

    else:
        logging.info("Tradeline data post-customer data - investigate further")

    return


def load_and_combine_data(path: Path):
    """Function to load and combine data."""

    logger.info("Loading data...")

    # load and assign flag for which train (1) or test (0)
    att_train = pd.read_csv(f"{path}/attributes.csv").assign(which=1)
    loans_train = pd.read_csv(f"{path}/loans.csv").assign(which=1)
    trade_train = pd.read_csv(f"{path}/tradeline.csv").assign(which=1)

    logger.info(f"att_train is shape: {att_train.shape}")
    logger.info(f"loans_train is shape: {loans_train.shape}")
    logger.info(f"trade_train is shape: {trade_train.shape}")

    att_test = pd.read_csv(f"{path}/attributes_test.csv").assign(which=0)
    loans_test = pd.read_csv(f"{path}/loans_test.csv").assign(which=0)
    trade_test = pd.read_csv(f"{path}/tradeline_test.csv").assign(which=0)

    logger.info(f"att_test is shape: {att_test.shape}")
    logger.info(f"loans_test is shape: {loans_test.shape}")
    logger.info(f"trade_test is shape: {trade_test.shape}")

    # concatenate tradline data
    trade = pd.concat([trade_train, trade_test])

    logging.info("Creating customer dataset...")

    # create customer dataset, merging att and loans
    cust = pd.merge(
        pd.concat([att_train, att_test]), pd.concat([loans_train, loans_test])
    )

    logger.info(f"Customer dataset is shape: {cust.shape}")

    return cust, trade


def check_leakage(df: pd.DataFrame):
    """Function to check if loan_ids from the training set appear in the test set."""

    train_cust_list = df[df["which"] == 1]["loan_id"].to_list()

    if df[(df["which"] == 0) & (df["loan_id"].isin(train_cust_list))].shape[0] == 0:
        print("No customer leakage")

    else:
        print("Leakage")


def check_for_missing_data(df: pd.DataFrame):
    """Function to check all dataframe columns for NA values."""

    # check for blanks
    blanks = df.isna().sum()

    if blanks.sum() == 0:
        print("No missing values")
    else:
        for a in blanks:
            if a > 0:
                print("there are blanks in {}".format(a))
            else:
                pass


def output_split_clean_data(df: pd.DataFrame, save_path: Path):
    """ "Function to save cleaned train and test data."""

    logging.info("Saving cleaned data")

    # split back into train and test sets
    df[df["which"] == 1].drop(columns=["which"]).to_csv(
        f"{save_path}/customer_train.csv", index=False
    )
    df[df["which"] == 0].drop(columns=["which", "early_settled"]).to_csv(
        f"{save_path}/customer_test.csv", index=False
    )


def create_train_test_split(df: pd.DataFrame):
    """Function to create train and test datasets."""

    # get X and y
    X = df.drop(columns="early_settled")
    y = df["early_settled"]

    # train test splits of 0.25 in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=40
    )

    return X_train, X_test, y_train, y_test


def prep_datasets_for_tabnet(save_path: Path, X_train, y_train, X_test, y_test):
    """Function to prepare 3 datasets for TabNet."""

    logging.info("Prepping datasets for deep learning...")
    cust_val = pd.read_csv(f"{save_path}/customer_test.csv").drop(
        columns=["origination_date", "loan_id"]
    )

    # create arbitrary column for processing reasons
    cust_val["early_settled"] = 3

    X_val = cust_val.drop(columns="early_settled")
    y_val = cust_val["early_settled"]

    d = dict.fromkeys(X_train.select_dtypes(np.int64).columns, np.int32)
    e = dict.fromkeys(X_train.select_dtypes("float64").columns, np.int32)

    X_train = X_train.astype(d).astype(e)
    X_test = X_test.astype(d).astype(e)
    X_val = X_val.astype(d).astype(e)

    logging.info("Saving deep learning datasets...")
    pd.concat([X_train, y_train.astype(int)], axis=1).to_csv(
        "./data/train.csv", index=False
    )
    pd.concat([X_test, y_test.astype(int)], axis=1).to_csv(
        "./data/test.csv", index=False
    )
    pd.concat([X_val, y_val.astype(int)], axis=1).to_csv("./data/val.csv", index=False)

    return
