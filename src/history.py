"""Validation for one student's ordered course-attempt history."""

from __future__ import annotations

from math import isfinite
from typing import Any

from src.config import COURSE_WEEKS, FORBIDDEN_MODEL_COLUMNS, HISTORY_REQUIRED_FIELDS


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _number(
    row: dict[str, Any],
    field: str,
    low: float,
    high: float,
    *,
    optional: bool = False,
) -> None:
    value = row[field]
    if _missing(value):
        if optional:
            return
        raise ValueError(f"Field '{field}' is required.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Field '{field}' must be numeric.")
    if not isfinite(float(value)) or not low <= float(value) <= high:
        raise ValueError(f"Field '{field}' must be between {low} and {high}.")


def validate_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and return a defensive copy of an ordinary prediction history."""
    if not history:
        raise ValueError("Student history cannot be empty.")
    if not all(isinstance(row, dict) for row in history):
        raise ValueError("Every weekly history record must be an object.")

    rows = [dict(row) for row in history]
    forbidden_payload_fields = set(FORBIDDEN_MODEL_COLUMNS) - {"attempt_id"}
    for index, row in enumerate(rows, start=1):
        leaked = forbidden_payload_fields.intersection(row)
        if "final_exam_score_audit_only" in leaked:
            raise ValueError(
                "Field 'final_exam_score_audit_only' is audit-only and is rejected "
                "from ordinary early-warning prediction requests."
            )
        if leaked:
            raise ValueError(f"Targets or forbidden fields are not accepted: {sorted(leaked)}")
        missing = [field for field in HISTORY_REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Week record {index} is missing fields: {missing}")

    weeks = [row["week"] for row in rows]
    if any(isinstance(week, bool) or not isinstance(week, int) for week in weeks):
        raise ValueError("Week numbers must be integers.")
    if len(set(weeks)) != len(weeks):
        raise ValueError("Duplicate week numbers are not allowed.")
    if weeks != sorted(weeks):
        raise ValueError("Weekly records must be ordered chronologically.")
    if weeks != list(range(1, max(weeks) + 1)):
        raise ValueError("Week numbers must be consecutive and start at week 1.")
    if not 1 <= max(weeks) <= COURSE_WEEKS:
        raise ValueError(f"Week numbers must be between 1 and {COURSE_WEEKS}.")

    if len({row["course_id"] for row in rows}) != 1:
        raise ValueError("All history records must belong to the same course.")
    if len({row["semester"] for row in rows}) != 1:
        raise ValueError("All history records must belong to the same semester.")
    if len({row["attempt_id"] for row in rows}) != 1:
        raise ValueError("All history records must belong to the same course attempt.")

    midterms: list[float] = []
    for row in rows:
        _number(row, "weekly_grade", 0, 100)
        _number(row, "attendance", 0, 100)
        _number(row, "assignment_score", 0, 100, optional=True)
        _number(row, "quiz_score", 0, 100, optional=True)
        _number(row, "submission_delay_days", 0, 30)
        _number(row, "corrections_count", 0, 100)
        _number(row, "assignment_completed", 0, 1)
        _number(row, "quiz_completed", 0, 1)
        _number(row, "midterm_score", 0, 100, optional=True)
        if row["assignment_completed"] and _missing(row["assignment_score"]):
            raise ValueError("A completed assignment requires assignment_score.")
        if not row["assignment_completed"] and not _missing(row["assignment_score"]):
            raise ValueError("assignment_score requires assignment_completed=1.")
        if row["quiz_completed"] and _missing(row["quiz_score"]):
            raise ValueError("A completed quiz requires quiz_score.")
        if not row["quiz_completed"] and not _missing(row["quiz_score"]):
            raise ValueError("quiz_score requires quiz_completed=1.")
        if row["week"] < 7 and not _missing(row["midterm_score"]):
            raise ValueError("midterm_score cannot be supplied before week 7.")
        if row["week"] >= 7:
            if _missing(row["midterm_score"]):
                raise ValueError("midterm_score is required from week 7 onward.")
            midterms.append(float(row["midterm_score"]))

    if max(weeks) >= 7:
        before_midterm = [row for row in rows if row["week"] < 7]
        assignments = sum(int(row["assignment_completed"]) for row in before_midterm)
        quizzes = sum(int(row["quiz_completed"]) for row in before_midterm)
        if assignments < 1 or quizzes < 1:
            raise ValueError(
                "At least one completed assignment and one completed quiz are required "
                "before the midterm checkpoint."
            )
        if len({round(value, 8) for value in midterms}) != 1:
            raise ValueError("midterm_score must remain fixed from week 7 onward.")

    for period, period_rows in (
        ("before the midterm", [row for row in rows if row["week"] < 7]),
        ("after the midterm", [row for row in rows if row["week"] >= 7]),
    ):
        if sum(int(row["assignment_completed"]) for row in period_rows) > 1:
            raise ValueError(f"Only one completed assignment is allowed {period}.")
        if sum(int(row["quiz_completed"]) for row in period_rows) > 1:
            raise ValueError(f"Only one completed quiz is allowed {period}.")

    return rows
