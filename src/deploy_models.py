"""Copy selected trained models into a compatible SRG-Tracker web project."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.config import MODEL_FILENAMES, MODELS_DIR, SELECTION_FILENAME, TARGETS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-project", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    args = parser.parse_args()
    tasks = tuple(dict.fromkeys(args.tasks))
    target_models = args.web_project / "models"
    target_selection = target_models / SELECTION_FILENAME
    source_selection = MODELS_DIR / SELECTION_FILENAME
    if not target_selection.exists():
        raise FileNotFoundError(f"Web-project selection manifest not found: {target_selection}")
    if not source_selection.exists():
        raise FileNotFoundError(f"Training selection manifest not found: {source_selection}")
    source_manifest = json.loads(source_selection.read_text(encoding="utf-8"))
    target_manifest = json.loads(target_selection.read_text(encoding="utf-8"))
    for task in tasks:
        artifact = MODELS_DIR / MODEL_FILENAMES[task]
        details = source_manifest.get("tasks", {}).get(task)
        if not artifact.exists() or not details:
            raise FileNotFoundError(f"No trained artifact and manifest entry available for {task!r}.")
        shutil.copy2(artifact, target_models / artifact.name)
        target_manifest.setdefault("tasks", {})[task] = details
        target_manifest.setdefault("configured_models", {})[task] = details["model"]
    target_manifest["selection_frozen_before_test"] = bool(source_manifest.get("selection_frozen_before_test"))
    target_selection.write_text(json.dumps(target_manifest, indent=2), encoding="utf-8")
    print(f"Deployed {', '.join(tasks)} model(s) to {target_models}.")


if __name__ == "__main__":
    main()
