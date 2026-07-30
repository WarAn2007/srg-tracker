"""Contract, leakage, feature, and split tests for SRG-Tracker V2.2."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.config import (
    AGGREGATED_FEATURE_COLUMNS,
    FINAL_SCORE_WEIGHTS,
    FORBIDDEN_MODEL_COLUMNS,
    RAW_WEEKLY_FEATURE_COLUMNS,
)
from src.eda import check_student_disjoint_splits
from src.features import aggregate_history, build_sequence
from src.history import validate_history
from src.inference import predict_history
from src.labels import score_to_gpa
from src.preprocessing import history_from_group, load_split


def valid_history(last_week: int = 7) -> list[dict[str, object]]:
    """Return a compact valid history suitable for validator unit tests."""
    rows: list[dict[str, object]] = []
    for week in range(1, last_week + 1):
        row: dict[str, object] = {
            "attempt_id": "S0001-PY101-1",
            "course_id": "PY101",
            "semester": 1,
            "week": week,
            "weekly_grade": 60.0 + week,
            "attendance": 80.0,
            "assignment_score": 70.0 if week in {3, 6} else None,
            "quiz_score": 68.0 if week in {2, 5} else None,
            "submission_delay_days": 0.5,
            "corrections_count": 0,
            "assignment_completed": int(week in {3, 6}),
            "quiz_completed": int(week in {2, 5}),
            "midterm_score": 72.0 if week >= 7 else None,
        }
        rows.append(row)
    return rows


class LabelContractTests(unittest.TestCase):
    def test_gpa_boundaries(self) -> None:
        expected = {
            64.99: 0.00,
            65.00: 2.00,
            70.00: 2.49,
            75.00: 2.99,
            80.00: 3.49,
            90.00: 3.99,
            90.01: 4.00,
            95.00: 4.49,
            95.01: 4.50,
        }
        for score, gpa in expected.items():
            with self.subTest(score=score):
                self.assertEqual(score_to_gpa(score), gpa)

    def test_final_score_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(FINAL_SCORE_WEIGHTS.values()), 1.0, places=12)
        self.assertEqual(FINAL_SCORE_WEIGHTS["final_exam_score_audit_only"], 0.25)


class HistoryValidationTests(unittest.TestCase):
    def test_valid_week_four_and_week_seven_histories(self) -> None:
        self.assertEqual(validate_history(valid_history(4))[-1]["week"], 4)
        self.assertEqual(validate_history(valid_history(7))[-1]["week"], 7)

    def test_rejects_empty_duplicate_unordered_and_nonconsecutive_histories(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_history([])
        duplicate = valid_history(4)
        duplicate.append(duplicate[-1].copy())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_history(duplicate)
        unordered = valid_history(4)
        unordered[1], unordered[2] = unordered[2], unordered[1]
        with self.assertRaisesRegex(ValueError, "ordered"):
            validate_history(unordered)
        nonconsecutive = valid_history(4)
        nonconsecutive.pop(1)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            validate_history(nonconsecutive)

    def test_rejects_mixed_attempts_and_timing_violations(self) -> None:
        mixed = valid_history(4)
        mixed[-1]["course_id"] = "ML201"
        with self.assertRaisesRegex(ValueError, "same course"):
            validate_history(mixed)
        early_midterm = valid_history(4)
        early_midterm[-1]["midterm_score"] = 80.0
        with self.assertRaisesRegex(ValueError, "before week 7"):
            validate_history(early_midterm)
        missing_midterm = valid_history(7)
        missing_midterm[-1]["midterm_score"] = None
        with self.assertRaisesRegex(ValueError, "required from week 7"):
            validate_history(missing_midterm)
        leaked_exam = valid_history(4)
        leaked_exam[-1]["final_exam_score_audit_only"] = 80.0
        with self.assertRaisesRegex(ValueError, "audit-only"):
            validate_history(leaked_exam)

    def test_requires_assignment_and_quiz_before_midterm(self) -> None:
        no_assessments = valid_history(7)
        for row in no_assessments:
            row["assignment_score"] = None
            row["quiz_score"] = None
            row["assignment_completed"] = 0
            row["quiz_completed"] = 0
        with self.assertRaisesRegex(ValueError, "assignment.*quiz"):
            validate_history(no_assessments)


class FeatureContractTests(unittest.TestCase):
    def test_model_feature_lists_exclude_all_forbidden_columns(self) -> None:
        forbidden = set(FORBIDDEN_MODEL_COLUMNS)
        self.assertTrue(forbidden.isdisjoint(RAW_WEEKLY_FEATURE_COLUMNS))
        self.assertTrue(forbidden.isdisjoint(AGGREGATED_FEATURE_COLUMNS))

    def test_aggregate_does_not_read_future_week(self) -> None:
        history = valid_history(5)
        at_four = aggregate_history(history[:4])
        history[-1]["weekly_grade"] = 0.0
        still_at_four = aggregate_history(history[:4])
        self.assertEqual(at_four, still_at_four)

    def test_sequence_padding_has_explicit_mask(self) -> None:
        values, mask = build_sequence(valid_history(4), max_weeks=14)
        self.assertEqual(values.shape[0], 14)
        self.assertEqual(mask.shape, (14,))
        np.testing.assert_array_equal(mask[:4], np.ones(4, dtype=bool))
        np.testing.assert_array_equal(mask[4:], np.zeros(10, dtype=bool))
        self.assertTrue(np.all(values[4:] == 0))


class GeneratedSplitTests(unittest.TestCase):
    def test_student_splits_do_not_overlap(self) -> None:
        self.assertEqual(
            check_student_disjoint_splits(),
            {
                "train_validation": 0,
                "train_test": 0,
                "validation_test": 0,
            },
        )

    def test_saved_selection_produces_three_history_predictions(self) -> None:
        validation = load_split("validation")
        first_attempt = next(iter(validation.groupby("attempt_id")))[1]
        history = history_from_group(first_attempt, 7)
        result = predict_history(history)
        self.assertEqual(
            set(result),
            {
                "cutoff_week",
                "predicted_final_gpa",
                "predicted_course_outcome",
                "enroll_probability",
                "predicted_learning_pace",
                "advisory",
            },
        )
        self.assertGreaterEqual(result["predicted_final_gpa"], 0.0)
        self.assertLessEqual(result["predicted_final_gpa"], 4.5)


if __name__ == "__main__":
    unittest.main()
