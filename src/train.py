"""Train baseline and candidate pipelines, then save the validation winner per task."""

from __future__ import annotations

import argparse
from typing import Any

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from src.config import METRICS_DIR, MODELS_DIR, MODEL_FILENAMES, RANDOM_STATE, TARGETS
from src.evaluation import evaluate_model, metrics_table
from src.preprocessing import build_preprocessor, get_features, get_target, load_split
from src.utils import ensure_directories


def candidate_models(task: str) -> dict[str, Any]:
    """Return meaningful baseline and candidate estimators for a task."""
    if task == "score":
        return {
            "dummy_mean": DummyRegressor(strategy="mean"),
            "ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
            "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=150, random_state=RANDOM_STATE),
        }
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, random_state=RANDOM_STATE),
    }


def primary_metric(task: str) -> tuple[str, bool]:
    """Return validation metric and whether a larger value is better."""
    return {"score": ("rmse", False), "outcome": ("f1_enroll", True), "pace": ("macro_f1", True)}[task]


def build_pipeline(task: str, estimator: Any) -> Pipeline:
    """Combine preprocessing with one estimator; scaling is required for linear models."""
    scaled = isinstance(estimator, (Ridge, LogisticRegression))
    return Pipeline([("preprocessor", build_preprocessor(scale_numeric=scaled)), ("model", estimator)])


def train_task(
    task: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
) -> tuple[str, Pipeline, pd.DataFrame]:
    """Fit all candidates on train data and choose a winner using validation data."""
    train_x, train_y = get_features(train_frame), get_target(train_frame, task)
    validation_x, validation_y = get_features(validation_frame), get_target(validation_frame, task)
    records: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidate_models(task).items():
        pipeline = build_pipeline(task, estimator)
        pipeline.fit(train_x, train_y)
        metrics, _ = evaluate_model(task, pipeline, validation_x, validation_y)
        records.append({"task": task, "model": name, "split": "validation", **metrics})
        fitted[name] = pipeline
    metric, higher_is_better = primary_metric(task)
    winner = max(records, key=lambda row: row[metric]) if higher_is_better else min(records, key=lambda row: row[metric])
    winner_name = str(winner["model"])
    return winner_name, fitted[winner_name], pd.DataFrame(records)


def refit_selected_model(
    task: str,
    model_name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
) -> Pipeline:
    """Refit the selected candidate on train + validation before final testing."""
    combined = pd.concat([train_frame, validation_frame], ignore_index=True)
    pipeline = build_pipeline(task, candidate_models(task)[model_name])
    pipeline.fit(get_features(combined), get_target(combined, task))
    return pipeline


def main() -> None:
    """Run model selection for all tasks and write reusable artifacts/reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TARGETS) + ["all"], default="all")
    args = parser.parse_args()
    ensure_directories(MODELS_DIR, METRICS_DIR)
    train_frame, validation_frame = load_split("train"), load_split("validation")
    tasks = list(TARGETS) if args.task == "all" else [args.task]
    all_results: list[dict[str, Any]] = []
    for task in tasks:
        winner_name, _, results = train_task(task, train_frame, validation_frame)
        model = refit_selected_model(task, winner_name, train_frame, validation_frame)
        joblib.dump(model, MODELS_DIR / MODEL_FILENAMES[task])
        all_results.extend(results.to_dict("records"))
        print(
            f"Selected {winner_name} for {task}; refitted it on train + validation "
            f"and saved it to {MODELS_DIR / MODEL_FILENAMES[task]}"
        )
    report = metrics_table(all_results)
    report.to_csv(METRICS_DIR / "validation_model_comparison.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
