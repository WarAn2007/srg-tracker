# V4.0 prediction API contract

`POST /api/predict` accepts one JSON object with a `history` array. The history
represents one student course attempt from week 1 through the latest observed week.

## Weekly record

Every item requires:

```text
attempt_id
course_id
semester
week
weekly_grade
attendance
assignment_score
quiz_score
submission_delay_days
corrections_count
assignment_completed
quiz_completed
midterm_score
```

Use JSON `null` when an optional assessment did not occur. Scores and attendance
must be between 0 and 100. Completion fields must be `0` or `1`. Submission delay
must be between 0 and 30 days.

The public API rejects unknown fields.

## History rules

- The first item is week 1.
- Weeks are ordered and consecutive, with no duplicates or gaps.
- All records share the same attempt, course, and semester.
- Midterm score is `null` before week 7.
- Midterm score is required from week 7 onward and remains fixed.
- At least one assignment and one quiz must be completed before week 7.
- The V4.0 UI accepts optional assignment and quiz observations while preserving the same leakage-safe API contract.
- No target, future result, or completed-course outcome is a prediction input.

## Example

```json
{
  "history": [
    {
      "attempt_id": "S0001-PY101-1",
      "course_id": "PY101",
      "semester": 1,
      "week": 1,
      "weekly_grade": 72,
      "attendance": 90,
      "assignment_score": 74,
      "quiz_score": null,
      "submission_delay_days": 0,
      "corrections_count": 0,
      "assignment_completed": 1,
      "quiz_completed": 0,
      "midterm_score": null
    }
  ]
}
```

## Response

```text
student_id
course_id
semester
cutoff_week
predicted_final_gpa
predicted_course_outcome
enroll_probability
predicted_learning_pace
advisory
```

`enroll` means failing/repeating. All outputs require qualified instructor review.
They must not trigger automatic grading, enrolment decisions, or student ranking.

## Service endpoints

- `GET /api/health` reports whether the selected local model artifacts exist.
- `GET /api/model-info` reports the active model metadata for all three tasks.
- `POST /api/predict` validates history and returns predictions.
- `POST /api/models/{task}` validates and activates one trusted compatible `.joblib` artifact.
- `GET /docs` opens interactive Swagger documentation.
