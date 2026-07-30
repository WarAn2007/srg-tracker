"""Create reproducible EDA tables and figures from generated splits."""

from __future__ import annotations

import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR, TARGETS
from src.preprocessing import load_split
from src.utils import ensure_directories


def check_student_disjoint_splits() -> dict[str, int]:
    """Return pairwise student overlap counts; every value must be zero."""
    students = {
        split: set(load_split(split)["student_id"].unique())
        for split in ("train", "validation", "test")
    }
    return {
        "train_validation": len(students["train"] & students["validation"]),
        "train_test": len(students["train"] & students["test"]),
        "validation_test": len(students["validation"] & students["test"]),
    }


def split_overview() -> pd.DataFrame:
    """Summarize rows, students, attempts, courses, and weeks for all splits."""
    records: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        frame = load_split(split)
        records.append(
            {
                "split": split,
                "rows": len(frame),
                "students": frame["student_id"].nunique(),
                "attempts": frame["attempt_id"].nunique(),
                "courses": frame["course_id"].nunique(),
                "semesters": frame["semester"].nunique(),
                "weeks": frame["week"].nunique(),
            }
        )
    return pd.DataFrame(records)


def target_description(frame: pd.DataFrame) -> pd.DataFrame:
    """Return compact numeric and categorical target summaries."""
    numeric = (
        frame[["final_gpa", "final_course_score"]]
        .describe()
        .transpose()
        .reset_index(names="target")
    )
    categorical_rows: list[dict[str, object]] = []
    for target in ("course_outcome", "learning_pace"):
        counts = frame[target].value_counts()
        for label, count in counts.items():
            categorical_rows.append(
                {
                    "target": target,
                    "label": label,
                    "count": int(count),
                    "fraction": float(count / len(frame)),
                }
            )
    categorical = pd.DataFrame(categorical_rows)
    numeric.to_csv(METRICS_DIR / "target_numeric_describe.csv", index=False)
    categorical.to_csv(METRICS_DIR / "target_class_distribution.csv", index=False)
    return numeric


def create_eda_outputs() -> dict[str, pd.DataFrame]:
    """Write tabular EDA outputs and presentation-ready plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_directories(FIGURES_DIR, METRICS_DIR)
    train = load_split("train")
    overview = split_overview()
    overview.to_csv(METRICS_DIR / "eda_split_overview.csv", index=False)
    missing = (
        train.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing.to_csv(METRICS_DIR / "missing_values.csv", index=False)
    overlap = pd.Series(
        check_student_disjoint_splits(),
        name="overlapping_students",
    ).rename_axis("split_pair").reset_index()
    overlap.to_csv(METRICS_DIR / "student_split_overlap.csv", index=False)
    targets = target_description(train)

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.histplot(train[TARGETS["gpa"]], bins=24, color="#31688e", ax=axes[0])
    axes[0].set_title("Final GPA")
    sns.countplot(data=train, x=TARGETS["outcome"], color="#35b779", ax=axes[1])
    axes[1].set_title("Course outcome")
    sns.countplot(
        data=train,
        x=TARGETS["pace"],
        order=["behind", "on_track", "ahead"],
        color="#fde725",
        ax=axes[2],
    )
    axes[2].set_title("Learning pace")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "target_distributions.png", dpi=160)
    plt.close(figure)

    weekly = (
        train.groupby("week", as_index=False)
        .agg(
            average_grade=("weekly_grade", "mean"),
            attendance=("attendance", "mean"),
            assignment_completion=("assignment_completed", "mean"),
            quiz_completion=("quiz_completed", "mean"),
        )
    )
    weekly.to_csv(METRICS_DIR / "weekly_summary.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=weekly, x="week", y="average_grade", marker="o", ax=axis)
    sns.lineplot(data=weekly, x="week", y="attendance", marker="o", ax=axis)
    axis.set_title("Training indicators by week")
    axis.set_ylabel("Mean")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "weekly_indicators.png", dpi=160)
    plt.close(figure)
    return {
        "overview": overview,
        "missing": missing,
        "targets": targets,
        "overlap": overlap,
        "weekly": weekly,
    }


if __name__ == "__main__":
    outputs = create_eda_outputs()
    print(outputs["overview"].to_string(index=False))
