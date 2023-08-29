import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def return_validation_metrics(y_probs, y_true, model: str, save_path: Path):
    """Function that returns validation metrics for a model output.

    y_probs: predicted probabilities
    y_true: actual labels

    """

    logging.info("Returning validation metrics")

    false_positive_rate, true_positive_rate, threshold = roc_curve(y_true, y_probs)
    threshold = threshold[np.argmax(true_positive_rate - false_positive_rate)]

    predictions = (y_probs > threshold) * [1.0]

    model_performance_dict = {}

    # Prediction-based metrics
    model_performance_dict["accuracy"] = accuracy_score(y_true, predictions)
    model_performance_dict["f1_score"] = f1_score(y_true, predictions)
    model_performance_dict["sensitivity"] = recall_score(
        y_true, predictions, pos_label=1
    )
    model_performance_dict["specificity"] = recall_score(
        y_true, predictions, pos_label=0
    )
    model_performance_dict["pos_pred_value"] = precision_score(
        y_true, predictions, pos_label=1
    )
    model_performance_dict["neg_pred_value"] = precision_score(
        y_true, predictions, pos_label=0
    )

    # Score-based metrics
    model_performance_dict["average_precision"] = average_precision_score(
        y_true, y_probs
    )

    model_performance_dict["roc_auc"] = roc_auc_score(y_true, y_probs)

    # Composite metrics
    precision, recall, _ = precision_recall_curve(y_true, y_probs, pos_label=1)
    model_performance_dict["pr_auc"] = auc(recall, precision)

    # save results
    logging.info(f"Saving metrics for {model} model")

    with open(f"{save_path}/{model}_model_results.pkl", "wb") as handle:
        pickle.dump(model_performance_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return
