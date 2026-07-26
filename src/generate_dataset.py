"""Generate a deterministic, student-disjoint synthetic SRG-Tracker dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260726
WEEKS = 14
COURSES = ("PY101", "DS102", "ML201", "ST203", "AI204")
SPLIT_SIZES = {"train": 300, "validation": 100, "test": 50}


def clipped(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(np.clip(value, low, high))


def pace_label(current_score: float, attendance: float, momentum: float) -> str:
    """Define an interpretable weekly learning-pace label."""
    pace_index = 0.65 * current_score + 0.20 * attendance + 0.15 * momentum
    if pace_index < 55:
        return "behind"
    if pace_index < 65:
        return "on_track"
    return "ahead"


def generate_student_course(
    rng: np.random.Generator,
    student_id: str,
    split: str,
    semester: int,
    course_id: str,
    course_difficulty: float,
) -> list[dict[str, object]]:
    ability = rng.normal(73, 12)
    engagement = rng.normal(74, 13)
    attendance_tendency = rng.normal(0.82, 0.12)
    late_tendency = np.clip((70 - engagement) / 28, 0.02, 1.8)
    weekly_grades: list[float] = []
    assignment_scores: list[float] = []
    quiz_scores: list[float] = []
    delays: list[float] = []
    attendance_events: list[float] = []
    corrections = 0
    midterm_score: float | None = None
    rows: list[dict[str, object]] = []

    for week in range(1, WEEKS + 1):
        trend = (week - 7.5) * rng.normal(0.10, 0.07)
        weekly_effort = clipped(0.52 * ability + 0.48 * engagement - course_difficulty + trend + rng.normal(0, 9))
        attendance_events.append(float(rng.random() < np.clip(attendance_tendency + rng.normal(0, 0.05), 0.05, 0.99)))
        weekly_grades.append(clipped(weekly_effort + rng.normal(0, 7)))

        if week in (3, 6, 9, 12):
            assignment_scores.append(clipped(weekly_effort + rng.normal(0, 8)))
            delays.append(float(np.clip(rng.gamma(1.4, late_tendency), 0, 3)))
            if rng.random() < np.clip((66 - weekly_effort) / 120, 0.02, 0.35):
                corrections += int(rng.integers(1, 3))
        if week in (2, 5, 8, 11):
            quiz_scores.append(clipped(weekly_effort + rng.normal(0, 8)))
            delays.append(float(np.clip(rng.gamma(1.25, late_tendency), 0, 3)))
        if week == 7:
            midterm_score = clipped(0.55 * np.mean(weekly_grades) + 0.25 * np.mean(assignment_scores) + 0.20 * np.mean(quiz_scores) + rng.normal(0, 6))

        avg_weekly = float(np.mean(weekly_grades))
        avg_assignment = float(np.mean(assignment_scores)) if assignment_scores else np.nan
        avg_quiz = float(np.mean(quiz_scores)) if quiz_scores else np.nan
        attendance_rate = float(np.mean(attendance_events) * 100)
        momentum = weekly_grades[-1] - weekly_grades[max(0, len(weekly_grades) - 4)]
        rows.append(
            {
                "split": split,
                "student_id": student_id,
                "course_id": course_id,
                "semester": semester,
                "week": week,
                "attendance_rate_to_date": round(attendance_rate, 2),
                "average_weekly_grade_to_date": round(avg_weekly, 2),
                "average_assignment_score_to_date": round(avg_assignment, 2) if assignment_scores else np.nan,
                "average_quiz_score_to_date": round(avg_quiz, 2) if quiz_scores else np.nan,
                "mean_submission_delay_days_to_date": round(float(np.mean(delays)), 2) if delays else np.nan,
                "corrections_count_to_date": corrections,
                "assignments_completed_to_date": len(assignment_scores),
                "quizzes_completed_to_date": len(quiz_scores),
                "midterm_score_available": round(midterm_score, 2) if midterm_score is not None else np.nan,
                "learning_pace": pace_label(avg_weekly, attendance_rate, momentum),
            }
        )

    final_exam_score = clipped(0.45 * ability + 0.35 * engagement - 0.50 * course_difficulty + rng.normal(0, 9))
    final_course_score = clipped(
        0.25 * (midterm_score or 0)
        + 0.25 * final_exam_score
        + 0.10 * float(np.mean(attendance_events) * 100)
        + 0.10 * float(np.mean(weekly_grades))
        + 0.20 * float(np.mean(assignment_scores))
        + 0.20 * float(np.mean(quiz_scores))
    )
    outcome = "pass" if final_course_score >= 65 else "enroll"
    for row in rows:
        row["final_exam_score_audit_only"] = round(final_exam_score, 2)
        row["final_course_score"] = round(final_course_score, 2)
        row["course_outcome"] = outcome
    return rows


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    records: list[dict[str, object]] = []
    student_number = 1
    course_difficulty = dict(zip(COURSES, (3.0, 5.0, 9.0, 7.0, 11.0), strict=True))
    for split, size in SPLIT_SIZES.items():
        for _ in range(size):
            student_id = f"S{student_number:04d}"
            student_number += 1
            for semester in (1, 2):
                for course_id in COURSES:
                    records.extend(
                        generate_student_course(
                            rng, student_id, split, semester, course_id, course_difficulty[course_id]
                        )
                    )
    return pd.DataFrame.from_records(records)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    for split in SPLIT_SIZES:
        dataset.loc[dataset["split"] == split].drop(columns="split").to_csv(
            output_dir / f"{split}.csv", index=False
        )
    summary = (
        dataset.groupby("split", observed=True)
        .agg(rows=("student_id", "size"), students=("student_id", "nunique"), pass_rate=("course_outcome", lambda x: (x == "pass").mean()))
        .round(3)
    )
    summary.to_csv(output_dir / "split_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
