import logging

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing as prep
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


xgb_base_pipeline = Pipeline(
    [
        ("scaler", prep.StandardScaler()),
        ("xgb", xgb.XGBClassifier(random_state=42)),
    ]
)

lr_base_pipeline = Pipeline(
    [
        ("scaler", prep.StandardScaler()),
        (
            "lr",
            LogisticRegression(
                solver="liblinear", class_weight="balanced", random_state=0, n_jobs=-1
            ),
        ),
    ]
)


lr_parameters_grid = {
    "scaler": [StandardScaler()],
    "lr__penalty": [
        "l1",
        "l2",
    ],
    "lr__solver": ["lbfgs", "liblinear", "saga"],
    "lr__tol": [0.0001, 0.05],
    "lr__C": [1e-5, 1e-3, 1e-1, 1],
    "lr__warm_start": [True, False],
}

xgb_parameters_grid = {
    "scaler": [StandardScaler()],
    "xgb__n_estimators": [100, 800],  # Number of gradient boosted trees.
    "xgb__learning_rate": [
        0.01,
        0.1,
    ],  # Aka step-shrinkage. Lower = less accuracy & less chance of overfitting.
    "xgb__reg_lambda": [0, 5],  # L2 regularisation
    "xgb__max_depth": [4, 8],  # Maximum tree depth for base learners.
}


def get_best_baseline_pipeline(model_type: str, X: pd.DataFrame, y, model_save_path):
    """Function to return best estimator pipeline."""

    logging.info("Performing grid search...")

    # make log-loss scorer as grid-search expects a scoring parameter not a loss parameter
    LogLoss = make_scorer(
        log_loss, greater_is_better=False, needs_proba=True, labels=sorted(np.unique(y))
    )
    # define baseline dict
    baseline_dict = {
        "lr": {"pipeline": lr_base_pipeline, "params": lr_parameters_grid},
        "xgb": {"pipeline": xgb_base_pipeline, "params": xgb_parameters_grid},
    }

    model_dict = baseline_dict[model_type]

    # run grid search to return best_parameters
    grid = GridSearchCV(
        estimator=model_dict["pipeline"],
        param_grid=model_dict["params"],
        cv=StratifiedKFold(random_state=None),
        scoring=LogLoss,
        verbose=10,
    )

    grid_results = grid.fit(X, y)

    best_estimator = grid_results.best_estimator_

    logging.info("Saving best estimator")
    # save best estimator
    joblib.dump(best_estimator, f"{model_save_path}/{model_type}_pipeline.pkl")

    return best_estimator


def make_predictions(estimator: Pipeline, X: pd.DataFrame):
    """Function that makes predictions with a fitted pipeline."""

    logging.info("Returning predictions")

    y_score = estimator.predict_proba(X)[:, 1]

    return y_score
