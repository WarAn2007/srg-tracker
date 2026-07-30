"""Metrics, rankings, and operational measurements for SRG experiments."""

from __future__ import annotations

import io
import time
from typing import Any, Callable

import joblib
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

from src.config import TARGETS


def evaluate_predictions(
    task: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate the approved predictive metrics for one task."""
    if task not in TARGETS:
        raise ValueError(f"Unknown task '{task}'.")
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if task == "gpa":
        return {
            "mae": float(mean_absolute_error(true, predicted)),
            "rmse": float(mean_squared_error(true, predicted) ** 0.5),
            "r2": float(r2_score(true, predicted)),
            "within_0_25": float(np.mean(np.abs(true - predicted) <= 0.25)),
        }
    if task == "outcome":
        result = {
            "recall_enroll": float(
                recall_score(true, predicted, pos_label="enroll", zero_division=0)
            ),
            "f1_enroll": float(
                f1_score(true, predicted, pos_label="enroll", zero_division=0)
            ),
            "accuracy": float(accuracy_score(true, predicted)),
        }
        if y_probability is not None and len(np.unique(true)) == 2:
            result["roc_auc_enroll"] = float(
                roc_auc_score((true == "enroll").astype(int), y_probability)
            )
        return result
    labels = ["behind", "on_track", "ahead"]
    recalls = recall_score(true, predicted, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "accuracy": float(accuracy_score(true, predicted)),
        **{
            f"recall_{label}": float(value)
            for label, value in zip(labels, recalls, strict=True)
        },
    }


def serialized_size_bytes(model: Any) -> int:
    """Measure a joblib-serializable model without creating a report artifact."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.tell()


def median_inference_ms(
    predict_one: Callable[[int], Any],
    sample_count: int,
    *,
    warmups: int = 10,
    repeats: int = 100,
) -> float:
    """Measure median single-example inference time after warm-up."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    for index in range(min(warmups, sample_count)):
        predict_one(index)
    durations: list[float] = []
    for index in range(repeats):
        start = time.perf_counter()
        predict_one(index % sample_count)
        durations.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(durations))


def add_task_rankings(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank overall model rows in the correct direction for each task."""
    ranked = frame.copy()
    ranked["rank"] = np.nan
    rules = {
        "gpa": ("rmse", True),
        "outcome": ("f1_enroll", False),
        "pace": ("macro_f1", False),
    }
    for task, (metric, ascending) in rules.items():
        mask = (ranked["task"] == task) & (ranked["cutoff"].astype(str) == "all")
        order = ranked.loc[mask, metric].rank(method="min", ascending=ascending)
        ranked.loc[mask, "rank"] = order
    return ranked.sort_values(
        ["task", "cutoff", "rank", "model"],
        na_position="last",
    ).reset_index(drop=True)
