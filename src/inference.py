"""Load trained artifacts and produce validated predictions for one weekly record."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS, MODEL_FILENAMES, MODELS_DIR


def validate_record(record: dict[str, Any]) -> pd.DataFrame:
    """Validate and order one raw weekly record for pipeline inference."""
    missing = [column for column in FEATURE_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    unexpected = set(record).difference(FEATURE_COLUMNS)
    if unexpected:
        raise ValueError(f"Unsupported fields (do not send targets or audit fields): {sorted(unexpected)}")
    return pd.DataFrame([{column: record[column] for column in FEATURE_COLUMNS}])


def load_models(models_dir=MODELS_DIR) -> dict[str, Any]:
    """Load the three selected task pipelines from disk."""
    models: dict[str, Any] = {}
    for task, filename in MODEL_FILENAMES.items():
        path = models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run `python -m src.train` first.")
        models[task] = joblib.load(path)
    return models


def predict_record(record: dict[str, Any], models: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return score, outcome risk probability, pace, and a human-review advisory."""
    features = validate_record(record)
    models = load_models() if models is None else models
    score = float(models["score"].predict(features)[0])
    outcome = str(models["outcome"].predict(features)[0])
    outcome_classes = list(models["outcome"].classes_)
    enroll_probability = float(models["outcome"].predict_proba(features)[0, outcome_classes.index("enroll")])
    pace = str(models["pace"].predict(features)[0])
    advisory = "Review this student with an instructor; this is an advisory prediction, not an academic decision."
    if outcome == "enroll" or pace == "behind":
        advisory = "Consider early academic support and instructor review; do not use this prediction as an automatic decision."
    return {
        "predicted_final_course_score": round(score, 2),
        "predicted_course_outcome": outcome,
        "enroll_probability": round(enroll_probability, 4),
        "predicted_learning_pace": pace,
        "advisory": advisory,
    }
