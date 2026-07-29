"""Safety and smoke tests for preprocessing, splitting, and inference."""

from __future__ import annotations

import unittest

from src.config import FEATURE_COLUMNS, LEAKAGE_COLUMNS
from src.eda import check_student_disjoint_splits
from src.inference import predict_record, validate_record
from src.preprocessing import get_features, load_split


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = load_split("train")

    def test_features_exclude_identifiers_targets_and_audit_data(self) -> None:
        features = get_features(self.train)
        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertTrue(set(LEAKAGE_COLUMNS).isdisjoint(features.columns))
        self.assertNotIn("student_id", features.columns)

    def test_student_splits_do_not_overlap(self) -> None:
        self.assertEqual(check_student_disjoint_splits(), {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        })

    def test_validation_rejects_missing_fields(self) -> None:
        with self.assertRaises(ValueError):
            validate_record({"week": 3})

    def test_validation_rejects_target_fields(self) -> None:
        record = get_features(self.train).iloc[0].to_dict()
        record["final_course_score"] = 75
        with self.assertRaises(ValueError):
            validate_record(record)

    def test_saved_models_produce_three_predictions(self) -> None:
        record = get_features(self.train).iloc[0].to_dict()
        result = predict_record(record)
        self.assertEqual(
            set(result),
            {
                "predicted_final_course_score",
                "predicted_course_outcome",
                "enroll_probability",
                "predicted_learning_pace",
                "advisory",
            },
        )


if __name__ == "__main__":
    unittest.main()
