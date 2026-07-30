"""Run the single held-out V2.2 test evaluation after selection is frozen."""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "reports" / ".matplotlib-cache"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import (
    EVALUATION_CUTOFFS,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_FILENAMES,
    MODELS_DIR,
    PREDICTIONS_DIR,
    TARGETS,
    V1_FEATURE_COLUMNS,
)
from src.evaluation import evaluate_predictions
from src.gru import INDEX_TO_PACE, build_sequence_dataset, predict_gru
from src.inference import load_gru, load_selection
from src.preprocessing import (
    build_aggregated_examples,
    build_latest_week_examples,
    get_features,
    load_split,
)
from src.train import decode_prediction
from src.utils import ensure_directories, save_json


STATE_FILE = METRICS_DIR / "final_evaluation_state.json"


def predict_selected_models(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Predict every frozen task for a supplied split without selecting models."""
    selection = load_selection()
    aggregated = build_aggregated_examples(frame)
    features = get_features(aggregated)
    latest = build_latest_week_examples(frame)
    outputs: dict[str, dict[str, Any]] = {}
    gru_predictions: dict[str, np.ndarray] | None = None
    encoder = None
    gru_model = None

    for task, details in selection["tasks"].items():
        if details["model"] == "gru":
            if gru_predictions is None:
                gru_model, encoder = load_gru()
                sequences = build_sequence_dataset(frame, encoder)
                gru_predictions = predict_gru(gru_model, sequences)
            outputs[task] = {
                "prediction": gru_predictions[task],
                "probability": (
                    gru_predictions["outcome_probability"]
                    if task == "outcome"
                    else (
                        gru_predictions["pace_probability"]
                        if task == "pace"
                        else None
                    )
                ),
            }
            continue

        model = joblib.load(MODELS_DIR / MODEL_FILENAMES[task])
        model_features = (
            latest.loc[:, V1_FEATURE_COLUMNS]
            if details["representation"] == "latest_week"
            else features
        )
        prediction = decode_prediction(task, model.predict(model_features))
        probability = None
        if task in {"outcome", "pace"} and hasattr(model, "predict_proba"):
            probability = model.predict_proba(model_features)
            if task == "outcome":
                classes = [int(value) for value in model.classes_]
                probability = probability[:, classes.index(1)]
        outputs[task] = {
            "prediction": prediction,
            "probability": probability,
        }
    return aggregated, outputs


def _metrics_rows(
    examples: pd.DataFrame,
    outputs: dict[str, dict[str, Any]],
    *,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, target_column in TARGETS.items():
        true = examples[target_column].to_numpy()
        predicted = np.asarray(outputs[task]["prediction"])
        probability = outputs[task]["probability"]
        for cutoff in ("all", *EVALUATION_CUTOFFS):
            mask = (
                np.ones(len(examples), dtype=bool)
                if cutoff == "all"
                else examples["cutoff_week"].to_numpy() == cutoff
            )
            task_probability = None
            if task == "outcome" and probability is not None:
                task_probability = np.asarray(probability)[mask]
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "cutoff": cutoff,
                    **evaluate_predictions(
                        task,
                        true[mask],
                        predicted[mask],
                        task_probability,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _save_predictions(
    examples: pd.DataFrame,
    outputs: dict[str, dict[str, Any]],
) -> None:
    metadata = examples[
        ["student_id", "attempt_id", "course_id", "semester", "cutoff_week"]
    ]
    for task, target_column in TARGETS.items():
        prediction = metadata.copy()
        prediction["actual"] = examples[target_column].to_numpy()
        prediction["predicted"] = outputs[task]["prediction"]
        probability = outputs[task]["probability"]
        if task == "outcome" and probability is not None:
            prediction["enroll_probability"] = np.asarray(probability)
        if task == "pace" and probability is not None:
            matrix = np.asarray(probability)
            for index, label in INDEX_TO_PACE.items():
                prediction[f"{label}_probability"] = matrix[:, index]
        if task == "gpa":
            prediction["absolute_error"] = np.abs(
                prediction["actual"].astype(float) - prediction["predicted"].astype(float)
            )
        else:
            prediction["correct"] = prediction["actual"] == prediction["predicted"]
        prediction.to_csv(PREDICTIONS_DIR / f"{task}_test_predictions.csv", index=False)


def _save_error_breakdown(
    examples: pd.DataFrame,
    outputs: dict[str, dict[str, Any]],
) -> None:
    rows: list[dict[str, object]] = []
    for task, target_column in TARGETS.items():
        work = examples[["course_id", "cutoff_week", target_column]].copy()
        work["predicted"] = outputs[task]["prediction"]
        for (course, cutoff), group in work.groupby(["course_id", "cutoff_week"]):
            metrics = evaluate_predictions(
                task,
                group[target_column].to_numpy(),
                group["predicted"].to_numpy(),
            )
            rows.append(
                {
                    "task": task,
                    "course_id": course,
                    "cutoff": int(cutoff),
                    "examples": len(group),
                    **metrics,
                }
            )
    pd.DataFrame(rows).to_csv(METRICS_DIR / "test_error_by_course_and_cutoff.csv", index=False)


def _save_calibration(
    examples: pd.DataFrame,
    outputs: dict[str, dict[str, Any]],
) -> None:
    rows: list[dict[str, object]] = []
    outcome_probability = np.asarray(outputs["outcome"]["probability"])
    outcome_true = (examples["course_outcome"].to_numpy() == "enroll").astype(int)
    observed, predicted = calibration_curve(
        outcome_true,
        outcome_probability,
        n_bins=10,
        strategy="quantile",
    )
    for bin_index, (mean_predicted, fraction_positive) in enumerate(
        zip(predicted, observed, strict=True)
    ):
        rows.append(
            {
                "task": "outcome",
                "class": "enroll",
                "bin": bin_index,
                "mean_predicted_probability": mean_predicted,
                "observed_fraction": fraction_positive,
            }
        )

    pace_probability = np.asarray(outputs["pace"]["probability"])
    pace_true = examples["learning_pace"].to_numpy()
    for class_index, label in INDEX_TO_PACE.items():
        observed, predicted = calibration_curve(
            (pace_true == label).astype(int),
            pace_probability[:, class_index],
            n_bins=10,
            strategy="quantile",
        )
        for bin_index, (mean_predicted, fraction_positive) in enumerate(
            zip(predicted, observed, strict=True)
        ):
            rows.append(
                {
                    "task": "pace",
                    "class": label,
                    "bin": bin_index,
                    "mean_predicted_probability": mean_predicted,
                    "observed_fraction": fraction_positive,
                }
            )
    calibration = pd.DataFrame(rows)
    calibration.to_csv(METRICS_DIR / "classification_calibration.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for task, axis in zip(("outcome", "pace"), axes, strict=True):
        subset = calibration.loc[calibration["task"] == task]
        for label, group in subset.groupby("class"):
            axis.plot(
                group["mean_predicted_probability"],
                group["observed_fraction"],
                marker="o",
                label=label,
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="black")
        axis.set_title(f"{task.title()} probability calibration")
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed fraction")
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "classification_calibration.png", dpi=160)
    plt.close(figure)


def _save_figures(
    examples: pd.DataFrame,
    outputs: dict[str, dict[str, Any]],
    metrics: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(6, 6))
    sample = np.random.default_rng(20260730).choice(
        len(examples),
        size=min(2000, len(examples)),
        replace=False,
    )
    axis.scatter(
        examples["final_gpa"].to_numpy()[sample],
        np.asarray(outputs["gpa"]["prediction"])[sample],
        alpha=0.28,
    )
    axis.plot([0, 4.5], [0, 4.5], linestyle="--", color="black")
    axis.set(xlabel="Observed GPA", ylabel="Predicted GPA", title="Observed vs predicted GPA")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "gpa_observed_vs_predicted.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    definitions = (
        ("gpa", "rmse", "GPA RMSE"),
        ("outcome", "f1_enroll", "Outcome enroll F1"),
        ("pace", "macro_f1", "Pace macro-F1"),
    )
    for axis, (task, metric, title) in zip(axes, definitions, strict=True):
        subset = metrics.loc[
            (metrics["task"] == task) & (metrics["cutoff"].astype(str) != "all")
        ].copy()
        subset["cutoff"] = subset["cutoff"].astype(int)
        sns.lineplot(data=subset, x="cutoff", y=metric, marker="o", ax=axis)
        axis.set_title(title)
        axis.set_xticks(EVALUATION_CUTOFFS)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "test_performance_by_cutoff.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, task, labels in (
        (axes[0], "outcome", ["enroll", "pass"]),
        (axes[1], "pace", ["behind", "on_track", "ahead"]),
    ):
        ConfusionMatrixDisplay.from_predictions(
            examples[TARGETS[task]],
            outputs[task]["prediction"],
            labels=labels,
            normalize="true",
            values_format=".2f",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(f"{task.title()} normalized confusion matrix")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "test_confusion_matrices.png", dpi=160)
    plt.close(figure)


def validate_on_non_test_split() -> pd.DataFrame:
    """Smoke-test frozen batch inference on validation without touching test."""
    validation = load_split("validation")
    examples, outputs = predict_selected_models(validation)
    return _metrics_rows(examples, outputs, split="validation")


def evaluate_test_set() -> pd.DataFrame:
    """Evaluate the frozen selection once and write all final reports."""
    if STATE_FILE.exists():
        raise RuntimeError(
            f"Final test evaluation already completed: {STATE_FILE}. "
            "Do not rerun it during model selection."
        )
    ensure_directories(FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR)
    selection = load_selection()
    if not selection.get("selection_frozen_before_test"):
        raise RuntimeError("Selection must be frozen before reading the test split.")
    test = load_split("test")
    examples, outputs = predict_selected_models(test)
    metrics = _metrics_rows(examples, outputs, split="test")
    metrics.to_csv(METRICS_DIR / "final_test_metrics.csv", index=False)
    _save_predictions(examples, outputs)
    _save_error_breakdown(examples, outputs)
    _save_calibration(examples, outputs)
    _save_figures(examples, outputs, metrics)
    save_json(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        METRICS_DIR / "environment.json",
    )
    save_json(
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "selection_file": str(MODELS_DIR / "selection.json"),
            "test_examples": int(len(examples)),
            "test_attempts": int(examples["attempt_id"].nunique()),
            "rerun_guard": True,
        },
        STATE_FILE,
    )
    return metrics


if __name__ == "__main__":
    print(evaluate_test_set().to_string(index=False))
