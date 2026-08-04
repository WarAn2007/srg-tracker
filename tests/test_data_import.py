"""Tests for external-dataset validation and student-disjoint splitting."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data_import import split_by_student, validate_frame


def external_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(3):
        rows.append(
            {
                "student_id": f"S{index}",
                "attempt_id": f"S{index}-PY101-1",
                "course_id": "PY101",
                "semester": 1,
                "week": 1,
                "weekly_grade": 70.0,
                "attendance": 90.0,
                "assignment_score": 72.0,
                "quiz_score": None,
                "submission_delay_days": 0.0,
                "corrections_count": 0,
                "assignment_completed": 1,
                "quiz_completed": 0,
                "midterm_score": None,
                "final_gpa": 3.0,
            }
        )
    return pd.DataFrame(rows)


class DataImportTests(unittest.TestCase):
    def test_gpa_only_data_is_valid_and_student_disjoint(self) -> None:
        frame = validate_frame(external_rows(), ("gpa",))
        splits = split_by_student(frame, seed=7)
        student_sets = [set(split["student_id"]) for split in splits.values()]
        self.assertTrue(all(student_sets))
        self.assertFalse(student_sets[0] & student_sets[1])
        self.assertFalse(student_sets[0] & student_sets[2])
        self.assertFalse(student_sets[1] & student_sets[2])


if __name__ == "__main__":
    unittest.main()
