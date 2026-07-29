"""Small shared utilities with no project-specific modelling logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directories(*directories: Path) -> None:
    """Create output directories when they do not already exist."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save a UTF-8 JSON file with stable, readable formatting."""
    ensure_directories(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
