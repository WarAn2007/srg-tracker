"""Create reproducible EDA tables and figures from the training split."""

from __future__ import annotations

import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR, TARGETS
from src.preprocessing import load_split
from src.utils import ensure_directories


def check_student_disjoint_splits() -> dict[str, int]:
    """Return pairwise student overlap counts; all values must be zero."""
    students = {
        split: set(load_split(split)["student_id"].unique())
        for split in ("train", "validation", "test")
    }
    return {
        "train_validation": len(students["train"] & students["validation"]),
        "train_test": len(students["train"] & students["test"]),
        "validation_test": len(students["validation"] & students["test"]),
    }


def create_eda_outputs() -> None:
    """Write compact descriptive tables and presentation-ready plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_directories(FIGURES_DIR, METRICS_DIR)
    frame = load_split("train")

    overview = pd.DataFrame(
        {
            "rows": [len(frame)],
            "students": [frame["student_id"].nunique()],
            "courses": [frame["course_id"].nunique()],
            "semesters": [frame["semester"].nunique()],
            "weeks": [frame["week"].nunique()],
        }
    )
    overview.to_csv(METRICS_DIR / "eda_overview.csv", index=False)
    frame.isna().mean().sort_values(ascending=False).rename("missing_rate").to_csv(
        METRICS_DIR / "missing_values.csv"
    )
    pd.Series(check_student_disjoint_splits(), name="overlapping_students").to_csv(
        METRICS_DIR / "student_split_overlap.csv"
    )

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.histplot(frame[TARGETS["score"]], bins=25, ax=axes[0], color="#31688e")
    axes[0].set_title("Final course score")
    sns.countplot(data=frame, x=TARGETS["outcome"], ax=axes[1], color="#35b779")
    axes[1].set_title("Course outcome")
    sns.countplot(
        data=frame,
        x=TARGETS["pace"],
        order=["behind", "on_track", "ahead"],
        ax=axes[2],
        color="#fde725",
    )
    axes[2].set_title("Weekly learning pace")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "target_distributions.png", dpi=160)
    plt.close(figure)

    weekly = (
        frame.groupby("week", as_index=False)
        .agg(
            average_grade=("average_weekly_grade_to_date", "mean"),
            attendance=("attendance_rate_to_date", "mean"),
            enroll_rate=("course_outcome", lambda values: (values == "enroll").mean()),
        )
    )
    weekly.to_csv(METRICS_DIR / "weekly_summary.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=weekly, x="week", y="average_grade", marker="o", label="Average grade", ax=axis)
    sns.lineplot(data=weekly, x="week", y="attendance", marker="o", label="Attendance", ax=axis)
    axis.set_title("Training data indicators by week")
    axis.set_ylabel("Mean value")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "weekly_indicators.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    create_eda_outputs()
    print(f"EDA outputs saved to {METRICS_DIR} and {FIGURES_DIR}")
