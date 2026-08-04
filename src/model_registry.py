"""Typed, thread-safe registry for packaged scikit-learn pipelines."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import AGGREGATED_FEATURE_COLUMNS, MODEL_FILENAMES, MODELS_DIR, SELECTION_FILENAME


TASKS = ("gpa", "outcome", "pace")


class ModelRegistryError(RuntimeError):
    """A model artifact or manifest cannot be used safely by the application."""


@dataclass(frozen=True)
class ModelMetadata:
    task: str
    model_type: str
    validation_metric: str
    validation_value: float
    rank: int
    artifact: str
    artifact_available: bool
    selection_method: str


def _probe_frame() -> pd.DataFrame:
    values: dict[str, Any] = {column: 0.0 for column in AGGREGATED_FEATURE_COLUMNS}
    values.update(
        {
            "course_id": "PY101",
            "semester": 1,
            "cutoff_week": 1,
            "observed_weeks": 1,
            "weekly_grade_current": 72.0,
            "weekly_grade_mean": 72.0,
            "attendance_current": 90.0,
            "attendance_mean": 90.0,
        }
    )
    return pd.DataFrame([values], columns=AGGREGATED_FEATURE_COLUMNS)


class ModelRegistry:
    """Load once at startup and atomically replace trusted local artifacts."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self._lock = threading.RLock()
        self._models: dict[str, Any] = {}
        self._manifest: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return set(self._models) == set(TASKS)

    def load(self) -> None:
        with self._lock:
            manifest = self._read_manifest()
            loaded: dict[str, Any] = {}
            for task in TASKS:
                artifact = self.models_dir / MODEL_FILENAMES[task]
                if not artifact.is_file():
                    raise ModelRegistryError(f"Missing {task} artifact: {artifact.name}")
                try:
                    model = joblib.load(artifact)
                except Exception as error:  # joblib raises several format/version exceptions
                    raise ModelRegistryError(
                        f"Could not load {artifact.name}. Check the Python and scikit-learn versions: {error}"
                    ) from error
                self._validate_model(task, model)
                loaded[task] = model
            self._models = loaded
            self._manifest = manifest

    def get_model(self, task: str) -> Any:
        with self._lock:
            if task not in self._models:
                raise ModelRegistryError(f"Model task is not ready: {task}")
            return self._models[task]

    def model_info(self) -> list[dict[str, Any]]:
        with self._lock:
            tasks = self._manifest.get("tasks", {})
            information: list[dict[str, Any]] = []
            for task in TASKS:
                details = tasks.get(task, {})
                metadata = ModelMetadata(
                    task=task,
                    model_type=str(details.get("model", "unknown")),
                    validation_metric=str(details.get("primary_metric", "not_provided")),
                    validation_value=float(details.get("validation_value", 0.0)),
                    rank=int(details.get("validation_rank", 0)),
                    artifact=MODEL_FILENAMES[task],
                    artifact_available=(self.models_dir / MODEL_FILENAMES[task]).is_file(),
                    selection_method=str(
                        details.get(
                            "selection_method",
                            self._manifest.get("selection_method", "not_provided"),
                        )
                    ),
                )
                information.append(asdict(metadata))
            return information

    def replace_model(
        self,
        task: str,
        uploaded_path: Path,
        *,
        model_type: str,
        validation_metric: str,
        validation_value: float,
        rank: int,
    ) -> dict[str, Any]:
        if task not in TASKS:
            raise ModelRegistryError(f"Unsupported model task: {task}")
        if not model_type.strip() or not validation_metric.strip():
            raise ModelRegistryError("Model type and validation metric are required.")
        try:
            candidate = joblib.load(uploaded_path)
        except Exception as error:
            raise ModelRegistryError(
                "The uploaded file is not a compatible joblib artifact. Only load files from a trusted source."
            ) from error
        self._validate_model(task, candidate)

        with self._lock:
            manifest = self._read_manifest()
            details = dict(manifest["tasks"][task])
            details.update(
                {
                    "model": model_type.strip(),
                    "primary_metric": validation_metric.strip(),
                    "validation_value": float(validation_value),
                    "validation_rank": int(rank),
                    "selection_method": "user_uploaded_trusted_artifact",
                }
            )
            manifest["tasks"][task] = details
            manifest["configured_models"][task] = model_type.strip()
            manifest["selection_method"] = "user_uploaded_trusted_artifact"
            manifest["selection_frozen_before_test"] = False
            manifest["held_out_evaluation_status"] = "not_run_for_current_runtime_mapping"
            manifest["runtime_configuration_updated_at"] = datetime.now(timezone.utc).isoformat()
            details["artifact_sha256"] = hashlib.sha256(uploaded_path.read_bytes()).hexdigest()

            destination = self.models_dir / MODEL_FILENAMES[task]
            staged_artifact = destination.with_suffix(destination.suffix + ".staged")
            backup_artifact = destination.with_suffix(destination.suffix + ".backup")
            staged_manifest = self.models_dir / f"{SELECTION_FILENAME}.staged"
            shutil.copy2(uploaded_path, staged_artifact)
            staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            try:
                if destination.exists():
                    os.replace(destination, backup_artifact)
                os.replace(staged_artifact, destination)
                os.replace(staged_manifest, self.models_dir / SELECTION_FILENAME)
                self._models[task] = candidate
                self._manifest = manifest
                backup_artifact.unlink(missing_ok=True)
            except Exception:
                if backup_artifact.exists():
                    os.replace(backup_artifact, destination)
                staged_artifact.unlink(missing_ok=True)
                staged_manifest.unlink(missing_ok=True)
                raise
            return next(item for item in self.model_info() if item["task"] == task)

    def _read_manifest(self) -> dict[str, Any]:
        path = self.models_dir / SELECTION_FILENAME
        if not path.is_file():
            raise ModelRegistryError(f"Missing model manifest: {SELECTION_FILENAME}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ModelRegistryError(f"Model manifest is invalid: {error}") from error
        tasks = manifest.get("tasks")
        configured = manifest.get("configured_models")
        if not isinstance(tasks, dict) or not isinstance(configured, dict):
            raise ModelRegistryError("Model manifest must contain tasks and configured_models objects.")
        missing = [task for task in TASKS if task not in tasks]
        if missing:
            raise ModelRegistryError(f"Model manifest is missing tasks: {', '.join(missing)}")
        return manifest

    @staticmethod
    def _validate_model(task: str, model: Any) -> None:
        if not callable(getattr(model, "predict", None)):
            raise ModelRegistryError(f"The {task} artifact does not provide predict().")
        if task == "outcome" and not callable(getattr(model, "predict_proba", None)):
            raise ModelRegistryError("The outcome artifact must provide predict_proba().")
        try:
            raw = np.asarray(model.predict(_probe_frame())).reshape(-1)[0]
            if task == "gpa":
                float(raw)
            elif task == "outcome":
                label = str(raw) if isinstance(raw, str) else ("enroll" if int(raw) == 1 else "pass")
                if label not in {"pass", "enroll"}:
                    raise ValueError("expected pass/enroll labels")
                probabilities = np.asarray(model.predict_proba(_probe_frame()))
                if probabilities.shape[0] != 1:
                    raise ValueError("invalid probability shape")
            else:
                label = str(raw) if isinstance(raw, str) else {0: "behind", 1: "on_track", 2: "ahead"}[int(raw)]
                if label not in {"behind", "on_track", "ahead"}:
                    raise ValueError("expected behind/on_track/ahead labels")
        except Exception as error:
            raise ModelRegistryError(
                f"The {task} artifact is incompatible with the V4.0 aggregated-history contract: {error}"
            ) from error
