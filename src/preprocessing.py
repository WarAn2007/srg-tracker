"""Safe dataset loading and feature preprocessing for all SRG tasks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, DATA_DIR, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGETS


def load_split(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load one generated split and check that its required columns exist."""
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}. Run src/generate_dataset.py first.")
    frame = pd.read_csv(path)
    required = set(FEATURE_COLUMNS) | set(TARGETS.values()) | {"student_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def get_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only prediction-time features; audit and target fields never enter a model."""
    missing = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing feature columns: {sorted(missing)}")
    return frame.loc[:, FEATURE_COLUMNS].copy()


def get_target(frame: pd.DataFrame, task: str) -> pd.Series:
    """Return the selected task target, rejecting unknown task names."""
    if task not in TARGETS:
        raise ValueError(f"Unknown task '{task}'. Choose one of: {', '.join(TARGETS)}")
    return frame[TARGETS[task]].copy()


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Create a train-fitted preprocessing step for raw weekly records."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Dense output keeps the shared transformer compatible with
            # HistGradientBoosting, which does not accept sparse matrices.
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, NUMERIC_FEATURES), ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)],
        remainder="drop",
    )
