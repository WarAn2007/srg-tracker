"""Runtime paths and input schema for the pre-trained local application."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
COURSE_WEEKS = 14

FORBIDDEN_MODEL_COLUMNS = [
    "attempt_id",
    "final_exam_score_audit_only",
    "final_course_score",
    "final_gpa",
    "course_outcome",
    "learning_pace",
]

HISTORY_REQUIRED_FIELDS = [
    "attempt_id",
    "course_id",
    "semester",
    "week",
    "weekly_grade",
    "attendance",
    "assignment_score",
    "quiz_score",
    "submission_delay_days",
    "corrections_count",
    "assignment_completed",
    "quiz_completed",
    "midterm_score",
]

_STATISTICS = (
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
)
_AGGREGATED_SIGNALS = (
    "weekly_grade",
    "attendance",
    "assignment_score",
    "quiz_score",
    "submission_delay_days",
)

AGGREGATED_FEATURE_COLUMNS = [
    "course_id",
    "semester",
    "cutoff_week",
    "observed_weeks",
    *[
        f"{signal}_{statistic}"
        for signal in _AGGREGATED_SIGNALS
        for statistic in _STATISTICS
    ],
    "corrections_count_total",
    "assignments_completed_total",
    "quizzes_completed_total",
    "assignment_completion_rate",
    "quiz_completion_rate",
    "midterm_score_available",
    "midterm_score",
]

MODEL_FILENAMES = {
    "gpa": "gpa_model.joblib",
    "outcome": "outcome_model.joblib",
    "pace": "pace_model.joblib",
}

SELECTION_FILENAME = "selection.json"
