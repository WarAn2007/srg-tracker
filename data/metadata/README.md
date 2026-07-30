# SRG-Tracker V2.2 synthetic dataset card

## Purpose

This synthetic dataset supports a history-aware student early-warning capstone.
It is designed to test leakage prevention, history representations, model
comparison, and reproducible evaluation. It is not evidence about real students
and is unsuitable for real academic decisions.

## Generation

- deterministic seed: `20260730`;
- 450 students;
- two semesters per student;
- five courses per semester;
- fourteen weekly records per course attempt;
- 4,500 attempts and 63,000 weekly rows;
- 300/100/50 student-disjoint train/validation/test split.

Stable latent traits include preparation, engagement, consistency, and
punctuality. Course difficulty, strictness, semester effects, workload, temporal
noise, improvement/decline trajectories, and occasional disruptions create
plausible but imperfect correlations.

## Unit of observation

One row is one student-course-semester-attempt at the end of one week.
`attempt_id` and `student_id` are identifiers and never model features.

## Raw weekly observations

- `weekly_grade`, `attendance`;
- `assignment_score`, `quiz_score`;
- `submission_delay_days`, `corrections_count`;
- `assignment_completed`, `quiz_completed`;
- `midterm_score`, unavailable before week 7.

The CSV also retains leakage-safe cumulative V1-compatible fields solely for the
frozen latest-week reference comparison. New tabular models derive their own
aggregates from raw history.

## Targets and audit fields

- `final_gpa`: 0.00-4.50, deterministically converted from final score;
- `course_outcome`: `pass` when score is at least 65, otherwise `enroll`;
- `learning_pace`: `behind`, `on_track`, or `ahead`;
- `final_course_score`: label-construction/audit field, never a model input;
- `final_exam_score_audit_only`: present only at week 14 and never a model input.

The transparent final-score rule is:

```text
0.25 * final_exam_score_audit_only
+ 0.25 * midterm_score
+ 0.10 * attendance_component
+ 0.10 * average_weekly_grade
+ 0.15 * average_assignment_score
+ 0.15 * average_quiz_score
```

The weights sum to exactly 1.00. The earlier V1 formula summed to 1.10; it is not
used by V2.2.

## Timing rules

- every attempt has consecutive weeks 1-14;
- at least one assignment and quiz are completed before week 7;
- midterm is missing before week 7 and fixed from week 7 onward;
- final exam audit data is missing before week 14;
- no future value is used to fill an earlier missing observation.

## Known limitations

The generator encodes simplified assumptions and threshold choices. Correlation
does not imply causation. It does not model institutional policy, accessibility,
demographics, socioeconomic context, curriculum changes, or intervention effects.
Performance on this dataset does not establish real-world validity, fairness, or
safety.
