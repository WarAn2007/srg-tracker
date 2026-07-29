# SRG-Tracker

**Student Risk and Grade Tracker** is an educational machine-learning capstone project. It uses weekly course activity to predict:

- final course score from 0 to 100;
- course outcome: `pass` or `enroll` (repeat the course);
- learning pace: `behind`, `on_track`, or `ahead`.

The first version uses reproducible synthetic data for 450 students. It is an experiment and demonstration only and must not be used for real academic decisions.

## First-version results

The selected model for every task is HistGradientBoosting. Selection used only the validation split. The selected pipelines were refitted on train + validation and evaluated once on the student-disjoint test split.

| Task | Test metrics |
| --- | --- |
| Final score | MAE 3.18, RMSE 4.26, R² 0.815 |
| Course outcome | enroll recall 0.720, enroll F1 0.759, ROC-AUC 0.941 |
| Learning pace | Macro-F1 0.855, balanced accuracy 0.851 |

Early-week score prediction is less accurate: test MAE falls from approximately 6.00 in week 1 to 2.08 in week 14 as more course information becomes available.

## Repository layout

```text
data/
  processed/          # student-disjoint train, validation and test CSV files
  metadata/           # dataset card and feature definitions
models/               # generated .joblib pipelines (ignored by Git)
notebooks/
  01_eda.ipynb
  02_baselines.ipynb
  03_demo.ipynb
  04_final_evaluation.ipynb
reports/
  figures/            # generated charts (ignored by Git)
  metrics/            # EDA, validation, test and error-analysis tables
  predictions/        # row-level held-out test predictions
src/                  # reusable generation, training, evaluation and inference code
tests/                # leakage, split, validation and inference checks
instructions.md       # detailed Russian-language learning guide
```

## Reproduce the project

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/generate_dataset.py
python -m src.eda
python -m src.train
python -m src.final_evaluation
python -m unittest discover -s tests -v
jupyter notebook
```

Open the notebooks in numeric order. `03_demo.ipynb` loads the saved pipelines and predicts all three targets for one weekly record.

## Leakage prevention

Only information available at the selected week enters the model. `student_id`, `final_exam_score_audit_only`, `final_course_score`, `course_outcome`, and all other target fields are excluded from model inputs. Split membership is based on unique students, so one student cannot appear in both training and evaluation data.

## Responsible use

The dataset is synthetic and encodes simplified assumptions about learning. It contains no real students or demographic attributes. Predictions are advisory indicators, not decisions about grades, enrolment, discipline, ranking, or access to education. A qualified instructor and an approved privacy/fairness process would be required before any real-world use.

## Author

Anvar Ibragimov
