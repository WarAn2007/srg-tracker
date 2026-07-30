"""Project-wide paths, schemas, targets, and reproducibility settings."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPORTS_DIR / ".matplotlib-cache"),
)

RANDOM_STATE = 20260730
COURSE_WEEKS = 14
EVALUATION_CUTOFFS = (4, 7, 10, 14)
COURSES = ("PY101", "DS102", "ML201", "ST203", "AI204")
SPLIT_SIZES = {"train": 300, "validation": 100, "test": 50}

TARGETS = {
    "gpa": "final_gpa",
    "outcome": "course_outcome",
    "pace": "learning_pace",
}

TASK_TYPES = {
    "gpa": "regression",
    "outcome": "classification",
    "pace": "classification",
}

FINAL_SCORE_WEIGHTS = {
    "final_exam_score_audit_only": 0.25,
    "midterm_score": 0.25,
    "attendance_component": 0.10,
    "average_weekly_grade": 0.10,
    "average_assignment_score": 0.15,
    "average_quiz_score": 0.15,
}

IDENTITY_COLUMNS = ["student_id", "attempt_id"]
FORBIDDEN_MODEL_COLUMNS = [
    "student_id",
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

RAW_WEEKLY_FEATURE_COLUMNS = [
    "weekly_grade",
    "attendance",
    "assignment_score",
    "assignment_score_available",
    "quiz_score",
    "quiz_score_available",
    "submission_delay_days",
    "corrections_count",
    "assignment_completed",
    "quiz_completed",
    "midterm_score",
    "midterm_score_available",
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

CATEGORICAL_FEATURES = ["course_id"]
NUMERIC_FEATURES = [
    column for column in AGGREGATED_FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES
]

V1_FEATURE_COLUMNS = [
    "course_id",
    "semester",
    "week",
    "attendance_rate_to_date",
    "average_weekly_grade_to_date",
    "average_assignment_score_to_date",
    "average_quiz_score_to_date",
    "mean_submission_delay_days_to_date",
    "corrections_count_to_date",
    "assignments_completed_to_date",
    "quizzes_completed_to_date",
    "midterm_score_available",
]

MODEL_FILENAMES = {
    "gpa": "gpa_model.joblib",
    "outcome": "outcome_model.joblib",
    "pace": "pace_model.joblib",
}

V1_REFERENCE_FILENAMES = {
    "gpa": "score_model_v1_reference.joblib",
    "outcome": "outcome_model_v1_reference.joblib",
    "pace": "pace_model_v1_reference.joblib",
}

GRU_MODEL_FILENAME = "gru_multitask.pt"
SELECTION_FILENAME = "selection.json"
