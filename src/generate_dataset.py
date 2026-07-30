"""Generate the deterministic, history-aware SRG-Tracker V2.2 dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    COURSE_WEEKS,
    COURSES,
    DATA_DIR,
    FINAL_SCORE_WEIGHTS,
    RANDOM_STATE,
    SPLIT_SIZES,
)
from src.labels import score_to_gpa
from src.utils import ensure_directories, save_json


ASSIGNMENT_WEEKS = {3, 6, 9, 12}
QUIZ_WEEKS = {2, 5, 8, 11}
COURSE_EFFECTS = {
    "PY101": (-1.0, 0.0),
    "DS102": (1.0, 1.5),
    "ML201": (5.0, 3.0),
    "ST203": (3.0, 2.0),
    "AI204": (6.0, 4.0),
}


@dataclass(frozen=True)
class StudentTraits:
    preparation: float
    engagement: float
    consistency: float
    punctuality: float


def clipped(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clip a numeric value to an approved range."""
    return float(np.clip(value, low, high))


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("At least one assessment is required before final label construction.")
    return float(np.mean(values))


def learning_pace_label(
    final_score: float,
    grade_trend: float,
    attendance: float,
    consistency: float,
) -> str:
    """Construct a final learning-pace label from transparent latent outcomes."""
    pace_index = (
        0.58 * final_score
        + 0.17 * attendance
        + 0.15 * clipped(50.0 + 7.0 * grade_trend)
        + 0.10 * consistency
    )
    if pace_index < 61.0:
        return "behind"
    if pace_index < 73.0:
        return "on_track"
    return "ahead"


def generate_attempt(
    rng: np.random.Generator,
    student_id: str,
    split: str,
    semester: int,
    course_id: str,
    traits: StudentTraits,
) -> list[dict[str, object]]:
    """Generate one complete 14-week course attempt."""
    difficulty, strictness = COURSE_EFFECTS[course_id]
    semester_effect = rng.normal(0.0, 2.0) + (semester - 1) * rng.normal(-0.3, 0.8)
    workload = rng.normal(0.0, 2.5)
    trajectory = rng.choice(
        np.asarray([-0.75, -0.35, 0.0, 0.30, 0.65]),
        p=np.asarray([0.10, 0.18, 0.30, 0.27, 0.15]),
    )
    disruption_week = int(rng.integers(3, 13)) if rng.random() < 0.20 else -1
    disruption_size = float(rng.uniform(5.0, 14.0))

    weekly_grades: list[float] = []
    attendance_values: list[float] = []
    assignment_scores: list[float] = []
    quiz_scores: list[float] = []
    delays: list[float] = []
    weekly_corrections: list[int] = []
    assignment_completed_values: list[int] = []
    quiz_completed_values: list[int] = []
    midterm_score: float | None = None
    rows: list[dict[str, object]] = []
    previous_noise = 0.0

    for week in range(1, COURSE_WEEKS + 1):
        temporal_noise = (
            0.42 * previous_noise
            + rng.normal(0.0, max(2.5, 9.5 - 0.07 * traits.consistency))
        )
        previous_noise = temporal_noise
        disruption = -disruption_size if week in {disruption_week, disruption_week + 1} else 0.0
        latent_performance = (
            0.48 * traits.preparation
            + 0.32 * traits.engagement
            + 0.20 * traits.consistency
            - difficulty
            - 0.35 * workload
            + semester_effect
            + trajectory * (week - 1)
            + disruption
            + temporal_noise
        )
        weekly_grade = clipped(latent_performance + rng.normal(0.0, 5.0))
        attendance_probability = np.clip(
            0.25
            + 0.0048 * traits.engagement
            + 0.0040 * traits.punctuality
            - 0.006 * difficulty
            + rng.normal(0.0, 0.04),
            0.20,
            0.99,
        )
        attendance = float(rng.random() < attendance_probability) * 100.0

        assignment_completed = 0
        assignment_score: float | None = None
        if week in ASSIGNMENT_WEEKS:
            forced = week == 3
            assignment_completed = int(
                forced
                or rng.random()
                < np.clip(0.35 + 0.004 * traits.engagement + 0.003 * traits.punctuality, 0.55, 0.98)
            )
            if assignment_completed:
                assignment_score = clipped(
                    latent_performance - 0.35 * strictness + rng.normal(0.0, 7.0)
                )
                assignment_scores.append(assignment_score)

        quiz_completed = 0
        quiz_score: float | None = None
        if week in QUIZ_WEEKS:
            forced = week == 2
            quiz_completed = int(
                forced
                or rng.random()
                < np.clip(0.40 + 0.004 * traits.engagement + 0.002 * traits.punctuality, 0.55, 0.99)
            )
            if quiz_completed:
                quiz_score = clipped(
                    latent_performance - 0.45 * strictness + rng.normal(0.0, 8.0)
                )
                quiz_scores.append(quiz_score)

        assessment_happened = bool(assignment_completed or quiz_completed)
        delay = (
            float(
                np.clip(
                    rng.gamma(
                        shape=1.25,
                        scale=max(0.08, (105.0 - traits.punctuality) / 42.0),
                    ),
                    0.0,
                    7.0,
                )
            )
            if assessment_happened
            else 0.0
        )
        correction_probability = np.clip((66.0 - weekly_grade) / 85.0, 0.02, 0.55)
        corrections = int(rng.integers(1, 3)) if rng.random() < correction_probability else 0

        weekly_grades.append(weekly_grade)
        attendance_values.append(attendance)
        delays.append(delay)
        weekly_corrections.append(corrections)
        assignment_completed_values.append(assignment_completed)
        quiz_completed_values.append(quiz_completed)

        if week == 7:
            midterm_score = clipped(
                0.48 * np.mean(weekly_grades)
                + 0.27 * _mean(assignment_scores)
                + 0.20 * _mean(quiz_scores)
                + 0.05 * traits.preparation
                - 0.25 * strictness
                + rng.normal(0.0, 5.0)
            )

        rows.append(
            {
                "split": split,
                "student_id": student_id,
                "attempt_id": f"{student_id}-{course_id}-{semester}",
                "course_id": course_id,
                "semester": semester,
                "week": week,
                "weekly_grade": round(weekly_grade, 2),
                "attendance": round(attendance, 2),
                "assignment_score": (
                    round(assignment_score, 2) if assignment_score is not None else np.nan
                ),
                "quiz_score": round(quiz_score, 2) if quiz_score is not None else np.nan,
                "submission_delay_days": round(delay, 2),
                "corrections_count": corrections,
                "assignment_completed": assignment_completed,
                "quiz_completed": quiz_completed,
                "midterm_score": (
                    round(midterm_score, 2) if midterm_score is not None else np.nan
                ),
                # V1-compatible latest-week reference features:
                "attendance_rate_to_date": round(float(np.mean(attendance_values)), 2),
                "average_weekly_grade_to_date": round(float(np.mean(weekly_grades)), 2),
                "average_assignment_score_to_date": (
                    round(float(np.mean(assignment_scores)), 2)
                    if assignment_scores
                    else np.nan
                ),
                "average_quiz_score_to_date": (
                    round(float(np.mean(quiz_scores)), 2) if quiz_scores else np.nan
                ),
                "mean_submission_delay_days_to_date": round(float(np.mean(delays)), 2),
                "corrections_count_to_date": int(sum(weekly_corrections)),
                "assignments_completed_to_date": int(sum(assignment_completed_values)),
                "quizzes_completed_to_date": int(sum(quiz_completed_values)),
                "midterm_score_available": (
                    round(midterm_score, 2) if midterm_score is not None else np.nan
                ),
            }
        )

    final_exam_score = clipped(
        0.36 * traits.preparation
        + 0.20 * traits.engagement
        + 0.14 * traits.consistency
        + 0.18 * np.mean(weekly_grades)
        + 0.07 * _mean(assignment_scores)
        + 0.05 * _mean(quiz_scores)
        - 0.55 * strictness
        + rng.normal(0.0, 7.5)
    )
    attendance_component = float(np.mean(attendance_values))
    final_course_score = clipped(
        FINAL_SCORE_WEIGHTS["final_exam_score_audit_only"] * final_exam_score
        + FINAL_SCORE_WEIGHTS["midterm_score"] * float(midterm_score)
        + FINAL_SCORE_WEIGHTS["attendance_component"] * attendance_component
        + FINAL_SCORE_WEIGHTS["average_weekly_grade"] * float(np.mean(weekly_grades))
        + FINAL_SCORE_WEIGHTS["average_assignment_score"] * _mean(assignment_scores)
        + FINAL_SCORE_WEIGHTS["average_quiz_score"] * _mean(quiz_scores)
        + rng.normal(0.0, 2.2)
    )
    outcome = "pass" if final_course_score >= 65.0 else "enroll"
    trend = float(np.polyfit(np.arange(COURSE_WEEKS), weekly_grades, 1)[0])
    pace = learning_pace_label(
        final_course_score,
        trend,
        attendance_component,
        traits.consistency,
    )
    gpa = score_to_gpa(final_course_score)

    for row in rows:
        row["final_exam_score_audit_only"] = (
            round(final_exam_score, 2) if row["week"] == COURSE_WEEKS else np.nan
        )
        row["final_course_score"] = round(final_course_score, 2)
        row["final_gpa"] = gpa
        row["course_outcome"] = outcome
        row["learning_pace"] = pace
    return rows


def build_dataset() -> pd.DataFrame:
    """Build all student-disjoint attempts and weekly records."""
    rng = np.random.default_rng(RANDOM_STATE)
    records: list[dict[str, object]] = []
    student_number = 1
    for split, size in SPLIT_SIZES.items():
        for _ in range(size):
            student_id = f"S{student_number:04d}"
            student_number += 1
            traits = StudentTraits(
                preparation=clipped(rng.normal(70.0, 13.0)),
                engagement=clipped(rng.normal(71.0, 15.0)),
                consistency=clipped(rng.normal(68.0, 14.0)),
                punctuality=clipped(rng.normal(73.0, 14.0)),
            )
            for semester in (1, 2):
                for course_id in COURSES:
                    records.extend(
                        generate_attempt(
                            rng,
                            student_id,
                            split,
                            semester,
                            course_id,
                            traits,
                        )
                    )
    return pd.DataFrame.from_records(records)


def validate_dataset(dataset: pd.DataFrame) -> dict[str, object]:
    """Validate generation invariants and return a concise quality report."""
    expected_rows = sum(SPLIT_SIZES.values()) * 2 * len(COURSES) * COURSE_WEEKS
    if len(dataset) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(dataset)}.")
    per_attempt = dataset.groupby("attempt_id")["week"].agg(["count", "min", "max"])
    if not (
        (per_attempt["count"] == COURSE_WEEKS)
        & (per_attempt["min"] == 1)
        & (per_attempt["max"] == COURSE_WEEKS)
    ).all():
        raise AssertionError("Every attempt must contain every week from 1 through 14.")
    if dataset.loc[dataset["week"] < 7, "midterm_score"].notna().any():
        raise AssertionError("midterm_score leaked before week 7.")
    if dataset.loc[dataset["week"] >= 7, "midterm_score"].isna().any():
        raise AssertionError("midterm_score is missing at or after week 7.")
    if dataset.loc[dataset["week"] < COURSE_WEEKS, "final_exam_score_audit_only"].notna().any():
        raise AssertionError("Final exam audit data leaked before course completion.")
    if dataset.loc[dataset["week"] == COURSE_WEEKS, "final_exam_score_audit_only"].isna().any():
        raise AssertionError("Final exam audit data is missing at course completion.")

    students = {
        split: set(dataset.loc[dataset["split"] == split, "student_id"])
        for split in SPLIT_SIZES
    }
    overlap = {
        "train_validation": len(students["train"] & students["validation"]),
        "train_test": len(students["train"] & students["test"]),
        "validation_test": len(students["validation"] & students["test"]),
    }
    if any(overlap.values()):
        raise AssertionError(f"Student overlap detected: {overlap}")

    attempt_level = dataset.loc[dataset["week"] == COURSE_WEEKS]
    correlations = (
        attempt_level[
            [
                "final_course_score",
                "average_weekly_grade_to_date",
                "attendance_rate_to_date",
                "average_assignment_score_to_date",
                "average_quiz_score_to_date",
            ]
        ]
        .corr()
        .round(3)
        .to_dict()
    )
    return {
        "seed": RANDOM_STATE,
        "rows": int(len(dataset)),
        "students": int(dataset["student_id"].nunique()),
        "attempts": int(dataset["attempt_id"].nunique()),
        "weeks_per_attempt": COURSE_WEEKS,
        "split_rows": {
            key: int(value)
            for key, value in dataset["split"].value_counts().sort_index().items()
        },
        "split_students": {
            key: int(value)
            for key, value in dataset.groupby("split")["student_id"].nunique().items()
        },
        "student_overlap": overlap,
        "target_ranges": {
            "final_course_score": [
                float(dataset["final_course_score"].min()),
                float(dataset["final_course_score"].max()),
            ],
            "final_gpa": [
                float(dataset["final_gpa"].min()),
                float(dataset["final_gpa"].max()),
            ],
        },
        "course_outcome_distribution": {
            str(key): int(value)
            for key, value in attempt_level["course_outcome"].value_counts().items()
        },
        "learning_pace_distribution": {
            str(key): int(value)
            for key, value in attempt_level["learning_pace"].value_counts().items()
        },
        "missingness_rules": {
            "midterm_missing_before_week_7": True,
            "midterm_present_from_week_7": True,
            "final_exam_present_only_at_week_14": True,
        },
        "attempt_level_correlations": correlations,
    }


def main() -> None:
    """Generate, validate, and save the three deterministic splits."""
    ensure_directories(DATA_DIR)
    dataset = build_dataset()
    quality_report = validate_dataset(dataset)
    for split in SPLIT_SIZES:
        dataset.loc[dataset["split"] == split].drop(columns="split").to_csv(
            DATA_DIR / f"{split}.csv",
            index=False,
        )
    summary = (
        dataset.groupby("split", observed=True)
        .agg(
            rows=("student_id", "size"),
            students=("student_id", "nunique"),
            attempts=("attempt_id", "nunique"),
            enroll_rate=("course_outcome", lambda values: (values == "enroll").mean()),
            average_gpa=("final_gpa", "mean"),
        )
        .round(4)
    )
    summary.to_csv(DATA_DIR / "split_summary.csv")
    save_json(quality_report, DATA_DIR / "data_quality_report.json")
    print(summary.to_string())


if __name__ == "__main__":
    main()
