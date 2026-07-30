# Full-history prediction input contract

`src.inference.predict_history(history)` accepts one ordered course attempt through
the latest submitted week. The history must start at week 1 and contain every week
through the cutoff.

## Weekly record

Every record requires:

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

Use JSON `null`/Python `None` for an assessment that did not occur. Scores and
attendance must be from 0 to 100. Completion fields must be 0 or 1. Delay must be
from 0 to 30 days.

## Valid week-4 example

```python
[
    {"attempt_id": "DEMO-PY101-1", "course_id": "PY101", "semester": 1,
     "week": 1, "weekly_grade": 68, "attendance": 100,
     "assignment_score": None, "quiz_score": None,
     "submission_delay_days": 0, "corrections_count": 0,
     "assignment_completed": 0, "quiz_completed": 0, "midterm_score": None},
    {"attempt_id": "DEMO-PY101-1", "course_id": "PY101", "semester": 1,
     "week": 2, "weekly_grade": 70, "attendance": 100,
     "assignment_score": None, "quiz_score": 69,
     "submission_delay_days": 0.2, "corrections_count": 0,
     "assignment_completed": 0, "quiz_completed": 1, "midterm_score": None},
    {"attempt_id": "DEMO-PY101-1", "course_id": "PY101", "semester": 1,
     "week": 3, "weekly_grade": 71, "attendance": 100,
     "assignment_score": 72, "quiz_score": None,
     "submission_delay_days": 0.4, "corrections_count": 0,
     "assignment_completed": 1, "quiz_completed": 0, "midterm_score": None},
    {"attempt_id": "DEMO-PY101-1", "course_id": "PY101", "semester": 1,
     "week": 4, "weekly_grade": 73, "attendance": 0,
     "assignment_score": None, "quiz_score": None,
     "submission_delay_days": 0, "corrections_count": 1,
     "assignment_completed": 0, "quiz_completed": 0, "midterm_score": None},
]
```

## Week 7

Extend the history consecutively through week 7. `midterm_score` must remain `None`
for weeks 1-6 and must be present at week 7. At least one assignment and one quiz
must have been completed before week 7.

## Week 14

Extend the same history through week 14. The ordinary prediction request still
must not include `final_exam_score_audit_only`. That field is reserved for a
separate completed-course audit and is removed before any model call.

## Rejected examples

These payloads raise field-level `ValueError` messages:

- empty history;
- duplicate, unordered, skipped, or out-of-range weeks;
- mixed `attempt_id`, course, or semester;
- missing required observation fields;
- midterm supplied before week 7;
- missing midterm from week 7 onward;
- no assignment or quiz before the midterm;
- scores outside 0-100;
- any target field;
- `final_exam_score_audit_only` in an ordinary prediction request.

## Output

The function returns:

```text
cutoff_week
predicted_final_gpa
predicted_course_outcome
enroll_probability
predicted_learning_pace
advisory
```

The advisory always states that synthetic-data predictions require human review.
