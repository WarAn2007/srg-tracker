"""Evaluate the frozen selected models once on the held-out test split."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_FILENAMES,
    MODELS_DIR,
    PREDICTIONS_DIR,
    TARGETS,
)
from src.evaluation import evaluate_model
from src.preprocessing import get_features, get_target, load_split
from src.utils import ensure_directories


def evaluate_test_set() -> pd.DataFrame:
    """Save final metrics, row predictions, and error slices by week/course."""
    ensure_directories(FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR)
    test = load_split("test")
    features = get_features(test)
    metric_rows: list[dict[str, object]] = []

    for task in TARGETS:
        model_path = MODELS_DIR / MODEL_FILENAMES[task]
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}; run `python -m src.train` first.")
        model = joblib.load(model_path)
        target = get_target(test, task)
        metrics, predictions = evaluate_model(task, model, features, target)
        metric_rows.append({"task": task, "split": "test", **metrics})

        prediction_frame = test[["student_id", "course_id", "semester", "week"]].copy()
        prediction_frame["actual"] = target
        prediction_frame["predicted"] = predictions
        if task == "score":
            prediction_frame["absolute_error"] = (
                prediction_frame["actual"] - prediction_frame["predicted"]
            ).abs()
        else:
            prediction_frame["correct"] = (
                prediction_frame["actual"] == prediction_frame["predicted"]
            )
        prediction_frame.to_csv(PREDICTIONS_DIR / f"{task}_test_predictions.csv", index=False)
        create_task_diagnostics(task, prediction_frame, target, predictions)

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(METRICS_DIR / "final_test_metrics.csv", index=False)
    return metrics_frame


def create_task_diagnostics(
    task: str,
    prediction_frame: pd.DataFrame,
    target: pd.Series,
    predictions,
) -> None:
    """Create week/course error tables and a suitable diagnostic figure."""
    if task == "score":
        weekly = prediction_frame.groupby("week", as_index=False)["absolute_error"].mean()
        course = prediction_frame.groupby("course_id", as_index=False)["absolute_error"].mean()
        weekly.to_csv(METRICS_DIR / "score_error_by_week.csv", index=False)
        course.to_csv(METRICS_DIR / "score_error_by_course.csv", index=False)

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.lineplot(data=weekly, x="week", y="absolute_error", marker="o", ax=axes[0])
        axes[0].set_title("Score MAE by week")
        sns.scatterplot(
            data=prediction_frame.sample(min(2000, len(prediction_frame)), random_state=20260726),
            x="actual",
            y="predicted",
            alpha=0.35,
            ax=axes[1],
        )
        axes[1].plot([0, 100], [0, 100], linestyle="--", color="black")
        axes[1].set_title("Actual vs predicted score")
    else:
        weekly = (
            prediction_frame.groupby("week", as_index=False)["correct"].mean()
            .rename(columns={"correct": "accuracy"})
        )
        course = (
            prediction_frame.groupby("course_id", as_index=False)["correct"].mean()
            .rename(columns={"correct": "accuracy"})
        )
        weekly.to_csv(METRICS_DIR / f"{task}_accuracy_by_week.csv", index=False)
        course.to_csv(METRICS_DIR / f"{task}_accuracy_by_course.csv", index=False)

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.lineplot(data=weekly, x="week", y="accuracy", marker="o", ax=axes[0])
        axes[0].set_ylim(0, 1.02)
        axes[0].set_title(f"{task.title()} accuracy by week")
        ConfusionMatrixDisplay.from_predictions(
            target,
            predictions,
            normalize="true",
            values_format=".2f",
            ax=axes[1],
            colorbar=False,
        )
        axes[1].set_title(f"{task.title()} normalized confusion matrix")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / f"{task}_test_diagnostics.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    print(evaluate_test_set().to_string(index=False))
