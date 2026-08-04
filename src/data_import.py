"""Import external weekly records and create leakage-safe student splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import DATA_DIR, HISTORY_REQUIRED_FIELDS, RANDOM_STATE, TARGETS
from src.history import validate_history
from src.utils import ensure_directories


SPLITS = ("train", "validation", "test")


def parse_tasks(values: Iterable[str]) -> tuple[str, ...]:
    tasks = tuple(dict.fromkeys(values))
    unknown = set(tasks).difference(TARGETS)
    if not tasks or unknown:
        raise ValueError(f"Tasks must be chosen from: {', '.join(TARGETS)}.")
    return tasks


def validate_frame(frame: pd.DataFrame, tasks: tuple[str, ...]) -> pd.DataFrame:
    """Validate external rows and return a normalized copy safe for training."""
    required = {"student_id", *HISTORY_REQUIRED_FIELDS, *(TARGETS[task] for task in tasks)}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input CSV must contain at least one row.")

    result = frame.copy()
    if result["student_id"].isna().any() or result["student_id"].astype(str).str.strip().eq("").any():
        raise ValueError("student_id is required for every row.")
    for attempt_id, group in result.groupby("attempt_id", sort=False):
        if group["student_id"].nunique(dropna=False) != 1:
            raise ValueError(f"Attempt {attempt_id!r} belongs to more than one student.")
        validate_history(group.loc[:, HISTORY_REQUIRED_FIELDS].sort_values("week").to_dict("records"))
        for task in tasks:
            target = TARGETS[task]
            if group[target].isna().any() or group[target].nunique(dropna=False) != 1:
                raise ValueError(f"Attempt {attempt_id!r} must have one non-empty {target!r} value.")

    if "gpa" in tasks and not result[TARGETS["gpa"]].between(0.0, 4.5).all():
        raise ValueError("final_gpa values must be between 0.0 and 4.5.")
    if "outcome" in tasks and not result[TARGETS["outcome"]].isin(["pass", "enroll"]).all():
        raise ValueError("course_outcome values must be 'pass' or 'enroll'.")
    if "pace" in tasks and not result[TARGETS["pace"]].isin(["behind", "on_track", "ahead"]).all():
        raise ValueError("learning_pace values must be behind, on_track, or ahead.")
    return result


def split_by_student(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Create deterministic 70/15/15 student-disjoint splits."""
    students = np.asarray(sorted(frame["student_id"].astype(str).unique()))
    if len(students) < 3:
        raise ValueError("At least three distinct students are required for train, validation, and test splits.")
    rng = np.random.default_rng(seed)
    rng.shuffle(students)
    validation_size = max(1, round(len(students) * 0.15))
    test_size = max(1, round(len(students) * 0.15))
    train_size = len(students) - validation_size - test_size
    if train_size < 1:
        raise ValueError("Not enough students to create three non-empty splits.")
    assignments = {
        "train": set(students[:train_size]),
        "validation": set(students[train_size : train_size + validation_size]),
        "test": set(students[train_size + validation_size :]),
    }
    return {
        split: frame.loc[frame["student_id"].astype(str).isin(students_for_split)].copy()
        for split, students_for_split in assignments.items()
    }


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path, tasks: tuple[str, ...]) -> None:
    """Write validated splits and concise, non-sensitive metadata."""
    ensure_directories(output_dir)
    student_sets = {name: set(frame["student_id"].astype(str)) for name, frame in splits.items()}
    if any(student_sets[left] & student_sets[right] for left in SPLITS for right in SPLITS if left < right):
        raise ValueError("Student overlap detected between supplied splits.")
    for split, frame in splits.items():
        frame.sort_values(["attempt_id", "week"]).to_csv(output_dir / f"{split}.csv", index=False)
    metadata = {
        "tasks": list(tasks),
        "splits": {name: {"rows": len(frame), "students": len(student_sets[name])} for name, frame in splits.items()},
        "student_disjoint": True,
    }
    (output_dir / "import_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="One full CSV; splits are created automatically.")
    source.add_argument("--train", type=Path, help="Pre-split train CSV; also requires --validation and --test.")
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--tasks", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks)
    if args.input:
        splits = split_by_student(validate_frame(pd.read_csv(args.input), tasks), args.seed)
    else:
        if not args.validation or not args.test:
            parser.error("--train requires --validation and --test.")
        splits = {
            "train": validate_frame(pd.read_csv(args.train), tasks),
            "validation": validate_frame(pd.read_csv(args.validation), tasks),
            "test": validate_frame(pd.read_csv(args.test), tasks),
        }
    save_splits(splits, args.output_dir, tasks)
    print(f"Imported {', '.join(tasks)} data into {args.output_dir}.")


if __name__ == "__main__":
    main()
