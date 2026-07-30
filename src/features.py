"""Leakage-safe aggregated and sequential history representations."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import (
    AGGREGATED_FEATURE_COLUMNS,
    RAW_WEEKLY_FEATURE_COLUMNS,
)
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


def _sequence_row(row: dict[str, Any]) -> list[float]:
    assignment_missing = row["assignment_score"] is None or (
        isinstance(row["assignment_score"], float) and np.isnan(row["assignment_score"])
    )
    quiz_missing = row["quiz_score"] is None or (
        isinstance(row["quiz_score"], float) and np.isnan(row["quiz_score"])
    )
    midterm_missing = row["midterm_score"] is None or (
        isinstance(row["midterm_score"], float) and np.isnan(row["midterm_score"])
    )
    values = {
        "weekly_grade": float(row["weekly_grade"]),
        "attendance": float(row["attendance"]),
        "assignment_score": 0.0 if assignment_missing else float(row["assignment_score"]),
        "assignment_score_available": float(not assignment_missing),
        "quiz_score": 0.0 if quiz_missing else float(row["quiz_score"]),
        "quiz_score_available": float(not quiz_missing),
        "submission_delay_days": float(row["submission_delay_days"]),
        "corrections_count": float(row["corrections_count"]),
        "assignment_completed": float(row["assignment_completed"]),
        "quiz_completed": float(row["quiz_completed"]),
        "midterm_score": 0.0 if midterm_missing else float(row["midterm_score"]),
        "midterm_score_available": float(not midterm_missing),
    }
    return [values[column] for column in RAW_WEEKLY_FEATURE_COLUMNS]


def build_sequence(
    history: list[dict[str, Any]],
    *,
    max_weeks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a raw weekly history and return values plus an explicit Boolean mask."""
    rows = validate_history(history)
    if len(rows) > max_weeks:
        raise ValueError(f"History contains {len(rows)} weeks; max_weeks is {max_weeks}.")
    values = np.zeros((max_weeks, len(RAW_WEEKLY_FEATURE_COLUMNS)), dtype=np.float32)
    mask = np.zeros(max_weeks, dtype=bool)
    for index, row in enumerate(rows):
        values[index] = np.asarray(_sequence_row(row), dtype=np.float32)
        mask[index] = True
    return values, mask
