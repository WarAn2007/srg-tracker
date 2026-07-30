"""Train and compare V1, tabular-history, MLP, and raw-sequence GRU models."""

from __future__ import annotations

import io
import json
import time
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    EVALUATION_CUTOFFS,
    GRU_MODEL_FILENAME,
    METRICS_DIR,
    MODEL_FILENAMES,
    MODELS_DIR,
    RANDOM_STATE,
    SELECTION_FILENAME,
    TARGETS,
    V1_FEATURE_COLUMNS,
    V1_REFERENCE_FILENAMES,
)
from src.evaluation import (
    add_task_rankings,
    evaluate_predictions,
    median_inference_ms,
    serialized_size_bytes,
)
from src.gru import (
    INDEX_TO_PACE,
    MultiTaskGRU,
    SequenceDataset,
    SequenceEncoder,
    build_sequence_dataset,
    predict_gru,
    train_gru,
)
from src.labels import score_to_gpa
from src.preprocessing import (
    build_aggregated_examples,
    build_latest_week_examples,
    build_preprocessor,
    get_features,
    load_split,
)
from src.utils import ensure_directories, save_json


CLASS_TO_INDEX = {
    "outcome": {"pass": 0, "enroll": 1},
    "pace": {"behind": 0, "on_track": 1, "ahead": 2},
}
INDEX_TO_CLASS = {
    task: {index: label for label, index in mapping.items()}
    for task, mapping in CLASS_TO_INDEX.items()
}


class V1GPAAdapter:
    """Adapt the frozen V1 final-score pipeline to the V2.2 GPA target."""

    def __init__(self, score_model: Any) -> None:
        self.score_model = score_model

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        scores = self.score_model.predict(features)
        return np.asarray([score_to_gpa(float(score)) for score in scores], dtype=float)


def candidate_estimators(task: str) -> dict[str, Any]:
    """Return all required tabular candidates with bounded CPU-friendly budgets."""
    if task == "gpa":
        return {
            "dummy": DummyRegressor(strategy="mean"),
            "linear": Ridge(alpha=1.0),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                max_iter=160,
                learning_rate=0.07,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
            "xgboost": XGBRegressor(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "catboost": CatBoostRegressor(
                iterations=180,
                depth=6,
                learning_rate=0.06,
                loss_function="RMSE",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
            "mlp_adam": MLPRegressor(
                hidden_layer_sizes=(64, 32),
                solver="adam",
                early_stopping=True,
                max_iter=220,
                random_state=RANDOM_STATE,
            ),
        }
    common_classifiers: dict[str, Any] = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "linear": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.07,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic" if task == "outcome" else "multi:softprob",
            eval_metric="logloss" if task == "outcome" else "mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=180,
            depth=6,
            learning_rate=0.06,
            loss_function="Logloss" if task == "outcome" else "MultiClass",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            auto_class_weights="Balanced",
        ),
        "mlp_adam": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            solver="adam",
            early_stopping=True,
            max_iter=220,
            random_state=RANDOM_STATE,
        ),
    }
    return common_classifiers


def build_pipeline(task: str, estimator: Any) -> Pipeline:
    """Combine train-fitted preprocessing with one estimator."""
    scale = isinstance(
        estimator,
        (Ridge, LogisticRegression, MLPRegressor, MLPClassifier),
    )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=scale)),
            ("model", estimator),
        ]
    )


def encode_target(task: str, values: pd.Series) -> np.ndarray:
    """Encode only classification labels; GPA remains continuous."""
    if task == "gpa":
        return values.to_numpy(dtype=float)
    return values.map(CLASS_TO_INDEX[task]).to_numpy(dtype=int)


def decode_prediction(task: str, values: np.ndarray) -> np.ndarray:
    """Decode estimator outputs to the public target vocabulary."""
    if task == "gpa":
        return np.clip(np.asarray(values, dtype=float).reshape(-1), 0.0, 4.5)
    flattened = np.asarray(values).reshape(-1)
    return np.asarray([INDEX_TO_CLASS[task][int(value)] for value in flattened])


def prediction_probability(task: str, model: Any, features: pd.DataFrame) -> np.ndarray | None:
    """Return enroll probability for outcome models."""
    if task != "outcome" or not hasattr(model, "predict_proba"):
        return None
    classes = [int(value) for value in model.classes_]
    return model.predict_proba(features)[:, classes.index(1)]


def _metric_records(
    task: str,
    model_name: str,
    representation: str,
    true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None,
    cutoffs: np.ndarray,
    *,
    train_seconds: float | None,
    inference_ms: float,
    model_size_bytes: int,
    parameter_count: int | None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for cutoff in ("all", *EVALUATION_CUTOFFS):
        mask = np.ones(len(true), dtype=bool) if cutoff == "all" else cutoffs == cutoff
        metrics = evaluate_predictions(
            task,
            true[mask],
            predicted[mask],
            probabilities[mask] if probabilities is not None else None,
        )
        records.append(
            {
                "task": task,
                "model": model_name,
                "representation": representation,
                "split": "validation",
                "cutoff": cutoff,
                "train_seconds": train_seconds,
                "median_inference_ms": inference_ms,
                "model_size_bytes": model_size_bytes,
                "parameter_count": parameter_count,
                **metrics,
            }
        )
    return records


def train_tabular_candidates(
    train_examples: pd.DataFrame,
    validation_examples: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], Pipeline]]:
    """Fit required aggregate-history candidates using train only."""
    train_x = get_features(train_examples)
    validation_x = get_features(validation_examples)
    cutoffs = validation_examples["cutoff_week"].to_numpy(dtype=int)
    records: list[dict[str, object]] = []
    fitted: dict[tuple[str, str], Pipeline] = {}
    for task, target_column in TARGETS.items():
        train_y = encode_target(task, train_examples[target_column])
        true = validation_examples[target_column].to_numpy()
        for name, estimator in candidate_estimators(task).items():
            pipeline = build_pipeline(task, estimator)
            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(train_x, train_y)
            train_seconds = time.perf_counter() - start
            predicted = decode_prediction(task, pipeline.predict(validation_x))
            probabilities = prediction_probability(task, pipeline, validation_x)
            inference_ms = median_inference_ms(
                lambda index, model=pipeline: model.predict(validation_x.iloc[[index]]),
                len(validation_x),
            )
            records.extend(
                _metric_records(
                    task,
                    name,
                    "aggregated_history",
                    true,
                    predicted,
                    probabilities,
                    cutoffs,
                    train_seconds=train_seconds,
                    inference_ms=inference_ms,
                    model_size_bytes=serialized_size_bytes(pipeline),
                    parameter_count=None,
                )
            )
            fitted[(task, name)] = pipeline
            print(f"Validated {task}/{name}")
    return records, fitted


def evaluate_v1_references(validation_frame: pd.DataFrame) -> list[dict[str, object]]:
    """Evaluate frozen V1 latest-week models on the new validation attempts."""
    examples = build_latest_week_examples(validation_frame)
    features = examples.loc[:, V1_FEATURE_COLUMNS]
    cutoffs = examples["week"].to_numpy(dtype=int)
    records: list[dict[str, object]] = []
    for task, filename in V1_REFERENCE_FILENAMES.items():
        path = MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing V1 reference artifact: {path}")
        model = joblib.load(path)
        if task == "gpa":
            raw = model.predict(features)
            predicted = np.asarray([score_to_gpa(float(value)) for value in raw])
            probability = None
        else:
            predicted = np.asarray(model.predict(features))
            probability = None
            if task == "outcome":
                classes = list(model.classes_)
                probability = model.predict_proba(features)[:, classes.index("enroll")]
        true = examples[TARGETS[task]].to_numpy()
        inference_ms = median_inference_ms(
            lambda index, fitted=model: fitted.predict(features.iloc[[index]]),
            len(features),
        )
        records.extend(
            _metric_records(
                task,
                "v1_reference",
                "latest_week",
                true,
                predicted,
                probability,
                cutoffs,
                train_seconds=None,
                inference_ms=inference_ms,
                model_size_bytes=path.stat().st_size,
                parameter_count=None,
            )
        )
    return records


def evaluate_gru_candidate(
    model: MultiTaskGRU,
    validation: SequenceDataset,
    training_seconds: float,
) -> tuple[list[dict[str, object]], int]:
    """Evaluate the shared raw-history GRU for all three heads."""
    predictions = predict_gru(model, validation)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size = buffer.tell()

    def predict_one(index: int) -> None:
        values = torch.from_numpy(validation.values[index : index + 1])
        masks = torch.from_numpy(validation.masks[index : index + 1])
        with torch.no_grad():
            model(values, masks)

    inference_ms = median_inference_ms(predict_one, len(validation.values))
    true_values = {
        "gpa": validation.gpa,
        "outcome": np.where(validation.outcome == 1, "enroll", "pass"),
        "pace": np.asarray([INDEX_TO_PACE[int(index)] for index in validation.pace]),
    }
    probability_values = {
        "gpa": None,
        "outcome": predictions["outcome_probability"],
        "pace": None,
    }
    records: list[dict[str, object]] = []
    for task in TARGETS:
        records.extend(
            _metric_records(
                task,
                "gru",
                "raw_sequence",
                true_values[task],
                predictions[task],
                probability_values[task],
                validation.cutoffs,
                train_seconds=training_seconds,
                inference_ms=inference_ms,
                model_size_bytes=size,
                parameter_count=model.parameter_count(),
            )
        )
    return records, size


def select_winners(comparison: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Freeze the best validation model per task."""
    overall = comparison.loc[comparison["cutoff"].astype(str) == "all"].copy()
    selection: dict[str, dict[str, object]] = {}
    for task, metric, ascending in (
        ("gpa", "rmse", True),
        ("outcome", "f1_enroll", False),
        ("pace", "macro_f1", False),
    ):
        rows = overall.loc[overall["task"] == task].sort_values(
            [metric, "model"],
            ascending=[ascending, True],
        )
        winner = rows.iloc[0]
        selection[task] = {
            "model": str(winner["model"]),
            "representation": str(winner["representation"]),
            "primary_metric": metric,
            "validation_value": float(winner[metric]),
        }
    return selection


def save_selected_artifacts(
    selection: dict[str, dict[str, object]],
    train_examples: pd.DataFrame,
    validation_examples: pd.DataFrame,
    gru_model: MultiTaskGRU,
    encoder: SequenceEncoder,
) -> None:
    """Refit selected tabular models on train+validation and save frozen artifacts."""
    combined = pd.concat([train_examples, validation_examples], ignore_index=True)
    combined_x = get_features(combined)
    for task, details in selection.items():
        model_name = str(details["model"])
        if model_name == "gru":
            joblib.dump(
                {
                    "backend": "gru",
                    "checkpoint": GRU_MODEL_FILENAME,
                    "task": task,
                },
                MODELS_DIR / MODEL_FILENAMES[task],
            )
            continue
        if model_name == "v1_reference":
            reference = joblib.load(MODELS_DIR / V1_REFERENCE_FILENAMES[task])
            artifact: Any = V1GPAAdapter(reference) if task == "gpa" else reference
        else:
            estimator = candidate_estimators(task)[model_name]
            artifact = build_pipeline(task, estimator)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                artifact.fit(
                    combined_x,
                    encode_target(task, combined[TARGETS[task]]),
                )
        joblib.dump(artifact, MODELS_DIR / MODEL_FILENAMES[task])

    checkpoint = {
        "state_dict": gru_model.state_dict(),
        "input_size": gru_model.input_size,
        "hidden_size": gru_model.hidden_size,
        "encoder": encoder.to_dict(),
        "parameter_count": gru_model.parameter_count(),
        "seed": RANDOM_STATE,
    }
    torch.save(checkpoint, MODELS_DIR / GRU_MODEL_FILENAME)
    save_json(
        {
            "selection_frozen_before_test": True,
            "seed": RANDOM_STATE,
            "cutoffs": list(EVALUATION_CUTOFFS),
            "tasks": selection,
        },
        MODELS_DIR / SELECTION_FILENAME,
    )


def write_comparison_reports(
    comparison: pd.DataFrame,
    selection: dict[str, dict[str, object]],
) -> None:
    """Save complete and task-specific ranked validation reports."""
    ranked = add_task_rankings(comparison)
    ranked.to_csv(METRICS_DIR / "validation_model_comparison.csv", index=False)
    for task in TARGETS:
        task_rows = ranked.loc[
            (ranked["task"] == task)
            & (ranked["cutoff"].astype(str) == "all")
        ].sort_values("rank")
        task_rows.to_csv(METRICS_DIR / f"{task}_model_ranking.csv", index=False)
    lines = [
        "# Validation model comparison",
        "",
        "Models were selected without reading test targets or test predictions.",
        "The V1 reference uses the latest weekly snapshot; tabular candidates use "
        "aggregated history; GRU uses the raw masked weekly sequence.",
        "",
        "## Frozen selection",
        "",
    ]
    for task, details in selection.items():
        lines.append(
            f"- **{task}**: `{details['model']}` using "
            f"`{details['representation']}`; {details['primary_metric']}="
            f"{details['validation_value']:.4f}."
        )
    lines.extend(
        [
            "",
            "Operational measurements use one validation history after 10 warm-up "
            "calls and report the median of 100 calls on this machine.",
        ]
    )
    (METRICS_DIR / "model_comparison_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run validation-only selection and save frozen production candidates."""
    final_state = METRICS_DIR / "final_evaluation_state.json"
    if final_state.exists():
        raise RuntimeError(
            f"Final test evaluation is already recorded in {final_state}. "
            "Do not retrain or change selection after inspecting held-out results. "
            "Start a new experiment with a new seed and test split instead."
        )
    ensure_directories(MODELS_DIR, METRICS_DIR)
    train_frame = load_split("train")
    validation_frame = load_split("validation")
    print("Building aggregated history examples...")
    train_examples = build_aggregated_examples(train_frame)
    validation_examples = build_aggregated_examples(validation_frame)

    records, _ = train_tabular_candidates(train_examples, validation_examples)
    records.extend(evaluate_v1_references(validation_frame))

    print("Building raw sequence examples...")
    encoder = SequenceEncoder.fit(train_frame)
    train_sequences = build_sequence_dataset(train_frame, encoder)
    validation_sequences = build_sequence_dataset(validation_frame, encoder)
    gru_model, gru_history, gru_seconds = train_gru(
        train_sequences,
        validation_sequences,
        encoder.input_size,
    )
    gru_records, _ = evaluate_gru_candidate(
        gru_model,
        validation_sequences,
        gru_seconds,
    )
    records.extend(gru_records)
    pd.DataFrame(gru_history).to_csv(
        METRICS_DIR / "gru_training_history.csv",
        index=False,
    )

    comparison = pd.DataFrame(records)
    selection = select_winners(comparison)
    save_selected_artifacts(
        selection,
        train_examples,
        validation_examples,
        gru_model,
        encoder,
    )
    write_comparison_reports(comparison, selection)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
