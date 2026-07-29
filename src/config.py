"""Project-wide paths, feature definitions, and reproducibility settings."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"

RANDOM_STATE = 20260726

TARGETS = {
    "score": "final_course_score",
    "outcome": "course_outcome",
    "pace": "learning_pace",
}

TASK_TYPES = {
    "score": "regression",
    "outcome": "classification",
    "pace": "classification",
}

ID_COLUMNS = ["student_id"]
LEAKAGE_COLUMNS = ["final_exam_score_audit_only", "final_course_score", "course_outcome"]

FEATURE_COLUMNS = [
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

CATEGORICAL_FEATURES = ["course_id"]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]

MODEL_FILENAMES = {task: f"{task}_model.joblib" for task in TARGETS}
