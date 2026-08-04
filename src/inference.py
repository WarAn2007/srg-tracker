"""Leakage-safe inference using the startup model registry."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import AGGREGATED_FEATURE_COLUMNS
from src.features import aggregate_history
from src.history import validate_history


def predict_history(history: list[dict[str, Any]], *, registry: Any) -> dict[str, Any]:
    """Validate a complete history and predict with already-loaded models."""
    rows = validate_history(history)
    aggregate = pd.DataFrame(
        [aggregate_history(rows)],
        columns=AGGREGATED_FEATURE_COLUMNS,
    )
    result: dict[str, Any] = {}

    for task in ("gpa", "outcome", "pace"):
        model = registry.get_model(task)
        raw = np.asarray(model.predict(aggregate)).reshape(-1)[0]
        if task == "gpa":
            prediction = float(np.clip(raw, 0.0, 4.5))
        elif task == "outcome":
            prediction = str(raw) if isinstance(raw, str) else ("enroll" if int(raw) == 1 else "pass")
            probabilities = model.predict_proba(aggregate)
            classes = list(model.classes_)
            positive = classes.index("enroll") if "enroll" in classes else classes.index(1)
            result["enroll_probability"] = round(float(probabilities[0, positive]), 4)
        else:
            prediction = str(raw) if isinstance(raw, str) else {0: "behind", 1: "on_track", 2: "ahead"}[int(raw)]
        result[task] = prediction

    outcome = str(result["outcome"])
    pace = str(result["pace"])
    advisory = (
        "Review this estimate with a qualified instructor. It is based on synthetic-data "
        "patterns and must not trigger an automatic academic decision."
    )
    if outcome == "enroll" or pace == "behind":
        advisory = (
            "Consider an early support conversation with a qualified instructor. This "
            "estimate must not trigger automatic grading, enrolment, or discipline."
        )

    return {
        "student_id": str(rows[0]["attempt_id"]),
        "course_id": str(rows[0]["course_id"]),
        "semester": int(rows[0]["semester"]),
        "cutoff_week": int(rows[-1]["week"]),
        "predicted_final_gpa": round(float(result["gpa"]), 2),
        "predicted_course_outcome": outcome,
        "enroll_probability": float(result.get("enroll_probability", np.nan)),
        "predicted_learning_pace": pace,
        "advisory": advisory,
    }
