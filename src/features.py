"""Leakage-safe aggregated history representation for the packaged models."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import AGGREGATED_FEATURE_COLUMNS
from src.history import validate_history


SIGNALS = (
    "weekly_grade",
    "attendance",
    "assignment_score",
    "quiz_score",
    "submission_delay_days",
)


def _present(values: list[Any]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in values if value is not None and not (isinstance(value, float) and np.isnan(value))],
        dtype=float,
    )


def _statistics(values: list[Any]) -> dict[str, float]:
    clean = _present(values)
    if clean.size == 0:
        return {name: np.nan for name in (
            "current",
            "mean",
            "median",
            "min",
            "max",
            "std",
            "last3_mean",
            "trend",
            "change_from_first",
            "volatility",
        )}
    trend = float(np.polyfit(np.arange(clean.size), clean, 1)[0]) if clean.size >= 2 else 0.0
    volatility = float(np.mean(np.abs(np.diff(clean)))) if clean.size >= 2 else 0.0
    return {
        "current": float(clean[-1]),
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "std": float(clean.std(ddof=0)),
        "last3_mean": float(clean[-3:].mean()),
        "trend": trend,
        "change_from_first": float(clean[-1] - clean[0]),
        "volatility": volatility,
    }


def aggregate_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce a validated history through its latest week to one tabular row."""
    rows = validate_history(history)
    result: dict[str, Any] = {
        "course_id": rows[0]["course_id"],
        "semester": int(rows[0]["semester"]),
        "cutoff_week": int(rows[-1]["week"]),
        "observed_weeks": len(rows),
    }
    for signal in SIGNALS:
        for statistic, value in _statistics([row[signal] for row in rows]).items():
            result[f"{signal}_{statistic}"] = value
    result.update(
        {
            "corrections_count_total": float(sum(float(row["corrections_count"]) for row in rows)),
            "assignments_completed_total": float(
                sum(float(row["assignment_completed"]) for row in rows)
            ),
            "quizzes_completed_total": float(
                sum(float(row["quiz_completed"]) for row in rows)
            ),
            "assignment_completion_rate": float(
                sum(float(row["assignment_completed"]) for row in rows) / len(rows)
            ),
            "quiz_completion_rate": float(
                sum(float(row["quiz_completed"]) for row in rows) / len(rows)
            ),
            "midterm_score_available": int(rows[-1]["week"] >= 7),
            "midterm_score": (
                float(rows[-1]["midterm_score"]) if rows[-1]["week"] >= 7 else np.nan
            ),
        }
    )
    return {column: result[column] for column in AGGREGATED_FEATURE_COLUMNS}
