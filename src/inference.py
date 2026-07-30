"""Validated full-history inference for the three frozen V2.2 tasks."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import (
    AGGREGATED_FEATURE_COLUMNS,
    GRU_MODEL_FILENAME,
    MODEL_FILENAMES,
    MODELS_DIR,
    SELECTION_FILENAME,
    V1_FEATURE_COLUMNS,
)
from src.features import aggregate_history
from src.gru import INDEX_TO_PACE, MultiTaskGRU, SequenceEncoder
from src.history import validate_history


def _v1_snapshot(history: list[dict[str, Any]]) -> pd.DataFrame:
    """Derive the frozen V1 snapshot contract from a validated raw history."""
    rows = validate_history(history)

    def values(field: str) -> list[float]:
        return [
            float(row[field])
            for row in rows
            if row[field] is not None
            and not (isinstance(row[field], float) and np.isnan(row[field]))
        ]

    snapshot = {
        "course_id": rows[0]["course_id"],
        "semester": int(rows[0]["semester"]),
        "week": int(rows[-1]["week"]),
        "attendance_rate_to_date": float(np.mean(values("attendance"))),
        "average_weekly_grade_to_date": float(np.mean(values("weekly_grade"))),
        "average_assignment_score_to_date": (
            float(np.mean(values("assignment_score")))
            if values("assignment_score")
            else np.nan
        ),
        "average_quiz_score_to_date": (
            float(np.mean(values("quiz_score"))) if values("quiz_score") else np.nan
        ),
        "mean_submission_delay_days_to_date": float(
            np.mean(values("submission_delay_days"))
        ),
        "corrections_count_to_date": int(
            sum(float(row["corrections_count"]) for row in rows)
        ),
        "assignments_completed_to_date": int(
            sum(float(row["assignment_completed"]) for row in rows)
        ),
        "quizzes_completed_to_date": int(
            sum(float(row["quiz_completed"]) for row in rows)
        ),
        "midterm_score_available": (
            float(rows[-1]["midterm_score"]) if rows[-1]["week"] >= 7 else np.nan
        ),
    }
    return pd.DataFrame([snapshot], columns=V1_FEATURE_COLUMNS)


def load_selection(models_dir=MODELS_DIR) -> dict[str, Any]:
    """Load and validate the frozen pre-test selection manifest."""
    path = models_dir / SELECTION_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Selection manifest not found: {path}. Run `python -m src.train`.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("selection_frozen_before_test"):
        raise ValueError("Model selection is not marked as frozen before test evaluation.")
    return payload


def load_gru(models_dir=MODELS_DIR) -> tuple[MultiTaskGRU, SequenceEncoder]:
    """Load the shared GRU and its train-fitted sequence encoder."""
    path = models_dir / GRU_MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"GRU checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = MultiTaskGRU(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    encoder = SequenceEncoder.from_dict(checkpoint["encoder"])
    return model, encoder


def _gru_prediction(
    history: list[dict[str, Any]],
    model: MultiTaskGRU,
    encoder: SequenceEncoder,
) -> dict[str, Any]:
    values, mask = encoder.transform_history(history)
    with torch.no_grad():
        gpa, outcome_logits, pace_logits = model(
            torch.from_numpy(values[None, :, :]),
            torch.from_numpy(mask[None, :]),
        )
    outcome_probability = float(torch.sigmoid(outcome_logits)[0])
    return {
        "gpa": float(gpa[0]),
        "outcome": "enroll" if outcome_probability >= 0.5 else "pass",
        "outcome_probability": outcome_probability,
        "pace": INDEX_TO_PACE[int(torch.argmax(pace_logits, dim=1)[0])],
    }


def predict_history(
    history: list[dict[str, Any]],
    *,
    models_dir=MODELS_DIR,
) -> dict[str, Any]:
    """Validate a complete history through its cutoff and return three predictions."""
    rows = validate_history(history)
    selection = load_selection(models_dir)
    aggregate = pd.DataFrame(
        [aggregate_history(rows)],
        columns=AGGREGATED_FEATURE_COLUMNS,
    )
    snapshot: pd.DataFrame | None = None
    gru_prediction: dict[str, Any] | None = None
    result: dict[str, Any] = {}

    for task, details in selection["tasks"].items():
        model_name = str(details["model"])
        if model_name == "gru":
            if gru_prediction is None:
                gru_model, encoder = load_gru(models_dir)
                gru_prediction = _gru_prediction(rows, gru_model, encoder)
            prediction = gru_prediction[task]
            if task == "outcome":
                result["enroll_probability"] = round(
                    float(gru_prediction["outcome_probability"]),
                    4,
                )
        else:
            path = models_dir / MODEL_FILENAMES[task]
            if not path.exists():
                raise FileNotFoundError(f"Selected model artifact not found: {path}")
            model = joblib.load(path)
            features = aggregate
            if details["representation"] == "latest_week":
                snapshot = _v1_snapshot(rows) if snapshot is None else snapshot
                features = snapshot
            raw = np.asarray(model.predict(features)).reshape(-1)[0]
            if task == "gpa":
                prediction = float(np.clip(raw, 0.0, 4.5))
            elif task == "outcome":
                prediction = (
                    str(raw)
                    if isinstance(raw, str)
                    else ("enroll" if int(raw) == 1 else "pass")
                )
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)
                    classes = list(model.classes_)
                    positive = (
                        classes.index("enroll")
                        if "enroll" in classes
                        else classes.index(1)
                    )
                    result["enroll_probability"] = round(
                        float(probabilities[0, positive]),
                        4,
                    )
            else:
                prediction = (
                    str(raw)
                    if isinstance(raw, str)
                    else {0: "behind", 1: "on_track", 2: "ahead"}[int(raw)]
                )
        result[task] = prediction

    outcome = str(result["outcome"])
    pace = str(result["pace"])
    advisory = (
        "Review the history with a qualified instructor. This synthetic-data "
        "prediction is advisory and must not trigger an automatic academic decision."
    )
    if outcome == "enroll" or pace == "behind":
        advisory = (
            "Consider early academic support and instructor review. Do not use this "
            "synthetic-data prediction for automatic grading or enrolment decisions."
        )
    return {
        "cutoff_week": int(rows[-1]["week"]),
        "predicted_final_gpa": round(float(result["gpa"]), 2),
        "predicted_course_outcome": outcome,
        "enroll_probability": float(result.get("enroll_probability", np.nan)),
        "predicted_learning_pace": pace,
        "advisory": advisory,
    }
