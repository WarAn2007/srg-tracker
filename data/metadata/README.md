# SRG-Tracker Synthetic Dataset Card

## Purpose

This synthetic dataset supports a student-risk and grade prediction capstone project. It is generated from transparent domain rules, not collected from real students. It is inspired by educational-activity concepts found in OULAD, but does not contain OULAD records or personal data.

## Unit of observation

One row represents a `student_id` in one `course_id`, during one `semester`, at the end of one `week` (1–14).

## Splits

The generated files contain disjoint students:

| Split | Students | Rows |
| --- | ---: | ---: |
| train | 300 | 42,000 |
| validation | 100 | 14,000 |
| test | 50 | 7,000 |

Each student has five courses in each of two semesters, with fourteen weekly snapshots.

## Features available at prediction time

- `week`, `semester`, `course_id`
- `attendance_rate_to_date`
- `average_weekly_grade_to_date`
- `average_assignment_score_to_date`
- `average_quiz_score_to_date`
- `mean_submission_delay_days_to_date`
- `corrections_count_to_date`
- `assignments_completed_to_date`, `quizzes_completed_to_date`
- `midterm_score_available` (missing before week 7)

`student_id` is an identifier for split and demo lookup, not a training feature. `final_exam_score` is retained only for auditability and must never be used as an input.

## Targets

- `final_course_score`: weighted final score (0–100).
- `course_outcome`: `pass` when final score ≥65; otherwise `enroll`.
- `learning_pace`: weekly supervised label: `behind`, `on_track`, or `ahead`.

## Final-score rule

`0.25 * midterm + 0.25 * final exam + 0.10 * attendance + 0.10 * weekly grade + 0.20 * assignments + 0.20 * quizzes`

## Limitations and responsible use

The generator intentionally simplifies student behaviour and may embed its own assumptions. The dataset has no demographic data, real academic records, or causal interpretation. It is unsuitable for production, student ranking, automatic enrolment decisions, or claims about real student performance.
