"""Metrics and diagnostic reports for SRG model experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.config import TARGETS, TASK_TYPES


def evaluate_predictions(task: str, y_true: pd.Series, y_pred: np.ndarray, y_probability: np.ndarray | None = None) -> dict[str, float]:
    """Calculate the agreed metrics for one task without fitting a model."""
    if task not in TARGETS:
        raise ValueError(f"Unknown task '{task}'.")
    if TASK_TYPES[task] == "regression":
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "r2": float(r2_score(y_true, y_pred)),
        }

    if task == "outcome":
        positive_label = "enroll"
        metrics = {
            "recall_enroll": float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
            "f1_enroll": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
        if y_probability is not None and y_true.nunique() == 2:
            metrics["roc_auc_enroll"] = float(roc_auc_score((y_true == positive_label).astype(int), y_probability))
        return metrics

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate_model(task: str, model: Any, features: pd.DataFrame, target: pd.Series) -> tuple[dict[str, float], np.ndarray]:
    """Predict with a fitted pipeline and return task metrics plus predictions."""
    predictions = model.predict(features)
    probabilities = None
    if task == "outcome" and hasattr(model, "predict_proba"):
        class_index = list(model.classes_).index("enroll")
        probabilities = model.predict_proba(features)[:, class_index]
    return evaluate_predictions(task, target, predictions, probabilities), predictions


def metrics_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert experiment records into a consistently ordered report table."""
    return pd.DataFrame(results).sort_values(["task", "model"]).reset_index(drop=True)
