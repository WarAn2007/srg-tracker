"""Release tests for the SRG-Tracker V4.0 local application."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from api.schemas import WeeklyRecord
from src.config import MODELS_DIR
from src.model_registry import ModelRegistry, ModelRegistryError


def history_payload() -> dict[str, object]:
    return {
        "history": [
            {
                "attempt_id": "S0001-PY101-1",
                "course_id": "PY101",
                "semester": 1,
                "week": 1,
                "weekly_grade": 72.0,
                "attendance": 90.0,
                "assignment_score": 74.0,
                "quiz_score": None,
                "submission_delay_days": 0.0,
                "corrections_count": 0,
                "assignment_completed": 1,
                "quiz_completed": 0,
                "midterm_score": None,
            }
        ]
    }


class RegistryTests(unittest.TestCase):
    def test_registry_loads_all_three_models_once(self) -> None:
        registry = ModelRegistry(MODELS_DIR)
        registry.load()
        self.assertTrue(registry.ready)
        self.assertEqual([item["task"] for item in registry.model_info()], ["gpa", "outcome", "pace"])

    def test_missing_artifact_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copy2(MODELS_DIR / "selection.json", target / "selection.json")
            with self.assertRaisesRegex(ModelRegistryError, "Missing gpa artifact"):
                ModelRegistry(target).load()

    def test_incompatible_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "selection.json").write_text(json.dumps({"tasks": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ModelRegistryError, "configured_models"):
                ModelRegistry(target).load()

    def test_one_task_can_be_replaced_without_reloading_other_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for filename in ("selection.json", "gpa_model.joblib", "outcome_model.joblib", "pace_model.joblib"):
                shutil.copy2(MODELS_DIR / filename, target / filename)
            registry = ModelRegistry(target)
            registry.load()
            outcome_before = registry.get_model("outcome")
            replacement = target / "replacement.joblib"
            shutil.copy2(target / "gpa_model.joblib", replacement)

            info = registry.replace_model(
                "gpa",
                replacement,
                model_type="trusted_test_model",
                validation_metric="rmse",
                validation_value=0.81,
                rank=2,
            )

            self.assertEqual(info["model_type"], "trusted_test_model")
            self.assertIs(registry.get_model("outcome"), outcome_before)
            manifest = json.loads((target / "selection.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["selection_frozen_before_test"])
            self.assertEqual(manifest["held_out_evaluation_status"], "not_run_for_current_runtime_mapping")
            self.assertEqual(len(manifest["tasks"]["gpa"]["artifact_sha256"]), 64)


class ApiTests(unittest.TestCase):
    def test_health_and_model_info_expose_registry_metadata(self) -> None:
        with TestClient(app) as client:
            health = client.get("/api/health")
            info = client.get("/api/model-info")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ready")
        self.assertTrue(health.json()["models_ready"])
        self.assertEqual(info.status_code, 200)
        self.assertEqual({item["task"] for item in info.json()}, {"gpa", "outcome", "pace"})
        self.assertTrue(all(item["artifact_available"] for item in info.json()))

    def test_valid_prediction_returns_identity_and_three_signals(self) -> None:
        with TestClient(app) as client:
            response = client.post("/api/predict", json=history_payload())
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["student_id"], "S0001-PY101-1")
        self.assertEqual(payload["course_id"], "PY101")
        self.assertEqual(payload["semester"], 1)
        self.assertIn(payload["predicted_course_outcome"], {"pass", "enroll"})
        self.assertIn(payload["predicted_learning_pace"], {"behind", "on_track", "ahead"})

    def test_leakage_field_is_rejected_with_structured_error(self) -> None:
        payload = history_payload()
        payload["history"][0]["final_exam_score_audit_only"] = 88  # type: ignore[index]
        with TestClient(app) as client:
            response = client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)
        error = response.json()["error"]
        self.assertEqual(error["code"], "request_validation_failed")
        self.assertTrue(error["fields"])

    def test_duplicate_assignment_in_one_period_is_rejected(self) -> None:
        payload = history_payload()
        duplicate = dict(payload["history"][0])  # type: ignore[index]
        duplicate.update({"week": 2, "weekly_grade": 75.0, "assignment_score": 81.0, "assignment_completed": 1})
        payload["history"].append(duplicate)  # type: ignore[union-attr]
        with TestClient(app) as client:
            response = client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only one completed assignment", response.json()["error"]["message"])

    def test_encrypted_history_is_written_to_local_history_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_dir = Path(directory) / "history"
            history_file = history_dir / "history.bin"
            with patch("api.main.HISTORY_DIR", history_dir), patch("api.main.HISTORY_FILE", history_file):
                with TestClient(app) as client:
                    saved = client.put("/api/history", content=b"SRGH\x01{}", headers={"Content-Type": "application/octet-stream"})
                    loaded = client.get("/api/history")
            self.assertEqual(saved.status_code, 204)
            self.assertEqual(loaded.status_code, 200)
            self.assertEqual(loaded.content, b"SRGH\x01{}")
            self.assertTrue(history_file.exists())


class SchemaTests(unittest.TestCase):
    def test_weekly_schema_exposes_only_observed_history(self) -> None:
        schema = WeeklyRecord.model_json_schema()
        self.assertNotIn("final_gpa", schema["properties"])
        self.assertNotIn("course_outcome", schema["properties"])
        self.assertFalse(schema.get("additionalProperties", True))


if __name__ == "__main__":
    unittest.main()
