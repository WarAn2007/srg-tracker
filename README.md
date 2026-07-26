# SRG-Tracker

**Student Risk and Grade Tracker** is an educational machine-learning capstone project. It uses weekly course activity to predict:

- final course score (0–100);
- course outcome: `pass` or `enroll` (repeat the course);
- learning pace: `behind`, `on_track`, or `ahead`.

## Project scope

The first version uses a reproducible synthetic dataset. It models 450 unique students across two semesters, five courses and fourteen weeks per course. It is designed for experimentation and demonstration only; it must not be used for real academic decisions.

The model receives only information available at the selected week. The final-exam score and final targets are never model inputs, preventing target leakage.

## Repository layout

```text
data/
  processed/       # train, validation and test CSV files
  metadata/        # data dictionary and dataset card
notebooks/         # EDA, modelling and final demo notebooks
src/               # reproducible Python source code
models/            # generated model artifacts (not committed)
reports/figures/   # generated charts (not committed)
docx/              # capstone brief and course requirements
```

## Dataset

Generate the dataset from the project root:

```powershell
python src/generate_dataset.py
```

The split is student-disjoint: 300 students in training, 100 in validation and 50 in test. Each row represents one student, course, semester and week.

The final course score is calculated as:

```text
0.25 × midterm + 0.25 × final_exam + 0.10 × attendance
+ 0.10 × average_weekly_grade + 0.20 × average_assignment
+ 0.20 × average_quiz
```

A score of 65 or higher is `pass`; otherwise the outcome is `enroll`.

## Planned modelling and evaluation

Three supervised models will be evaluated independently:

| Task | Target | Primary metrics |
| --- | --- | --- |
| Regression | `final_course_score` | MAE, RMSE, R² |
| Binary classification | `course_outcome` | Recall, F1, ROC-AUC |
| Multiclass classification | `learning_pace` | Macro F1, balanced accuracy |

The project will compare simple baselines with tree-based and linear models. The held-out test set will remain untouched until final evaluation.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/generate_dataset.py
```

## Responsible AI and limitations

This dataset is synthetic and encodes simplified assumptions about learning. Predictions are advisory indicators, not decisions about grades, enrolment, discipline or access to education. A qualified instructor must review any real-world intervention. The project intentionally excludes sensitive demographic attributes and student-to-student recommendations.

## Author

Anvar Ibragimov
