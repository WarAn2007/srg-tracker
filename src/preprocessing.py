"""Dataset loading and leakage-safe history preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    AGGREGATED_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    DATA_DIR,
    EVALUATION_CUTOFFS,
    HISTORY_REQUIRED_FIELDS,
    NUMERIC_FEATURES,
    TARGETS,
    V1_FEATURE_COLUMNS,
)
from src.features import aggregate_history


def load_split(
    split: str,
    data_dir: Path = DATA_DIR,
    tasks: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load one generated split and validate its minimum V2.2 schema."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}. Run `python -m src.generate_dataset` first."
        )
    frame = pd.read_csv(path)
    selected_tasks = tuple(TARGETS) if tasks is None else tuple(tasks)
    unknown = set(selected_tasks).difference(TARGETS)
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    required = (
        set(HISTORY_REQUIRED_FIELDS)
        | {TARGETS[task] for task in selected_tasks}
        | {"student_id"}
    )
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def history_from_group(group: pd.DataFrame, cutoff: int) -> list[dict[str, object]]:
    """Create a JSON-like prediction history through one cutoff."""
    selected = group.loc[group["week"] <= cutoff, HISTORY_REQUIRED_FIELDS].sort_values("week")
    if selected.empty or int(selected["week"].max()) != cutoff:
        raise ValueError(f"Attempt does not contain cutoff week {cutoff}.")
    records = selected.replace({np.nan: None}).to_dict("records")
    return [dict(record) for record in records]


def build_aggregated_examples(
    frame: pd.DataFrame,
    cutoffs: Iterable[int] = EVALUATION_CUTOFFS,
    tasks: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build one leakage-safe aggregate row per attempt and cutoff."""
    selected_tasks = tuple(TARGETS) if tasks is None else tuple(tasks)
    rows: list[dict[str, object]] = []
    for _, group in frame.groupby("attempt_id", sort=False):
        first = group.iloc[0]
        for cutoff in cutoffs:
            history = history_from_group(group, int(cutoff))
            row = aggregate_history(history)
            row.update(
                {
                    "student_id": first["student_id"],
                    "attempt_id": first["attempt_id"],
                    **{TARGETS[task]: first[TARGETS[task]] for task in selected_tasks},
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_latest_week_examples(
    frame: pd.DataFrame,
    cutoffs: Iterable[int] = EVALUATION_CUTOFFS,
) -> pd.DataFrame:
    """Return latest-week V1-compatible rows for the requested cutoffs."""
    selected = frame.loc[frame["week"].isin(tuple(cutoffs))].copy()
    columns = [
        "student_id",
        "attempt_id",
        *V1_FEATURE_COLUMNS,
        *TARGETS.values(),
        "final_course_score",
    ]
    return selected.loc[:, columns].reset_index(drop=True)


def get_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical aggregated feature frame."""
    missing = set(AGGREGATED_FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing aggregated features: {sorted(missing)}")
    return frame.loc[:, AGGREGATED_FEATURE_COLUMNS].copy()


def get_target(frame: pd.DataFrame, task: str) -> pd.Series:
    """Return one approved target."""
    if task not in TARGETS:
        raise ValueError(f"Unknown task '{task}'. Choose one of: {', '.join(TARGETS)}")
    return frame[TARGETS[task]].copy()


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Create train-fitted preprocessing for aggregated history features."""
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
