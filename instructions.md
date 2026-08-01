# Reproducing and using SRG-Tracker V2.3

V2.3 ships three ready-to-use, user-configured models:

- final GPA: MLP with Adam;
- course outcome: logistic regression;
- learning pace: logistic regression.

The repository also contains the validation comparison that records their ranks
and metrics. The selected models are a documented configuration choice, not a
claim that they are the automatic winners of every validation metric.

## 1. Prerequisites and environment

Use Windows PowerShell and Python 3.12. Verify the interpreter first:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins the direct package versions used for V2.3. The project
also fixes the random seed and CPU thread count in `src/config.py`.

If PowerShell blocks activation, run the commands through
`.\.venv\Scripts\python.exe`; activation is optional. If `py -3.12` is not
available, install Python 3.12 from python.org and reopen PowerShell.

## 2. Try the ready models

The repository contains the selected artifacts in `models/`. After installation,
start Jupyter and open the interactive product notebook:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```

Run `notebooks/final_product.ipynb` from top to bottom. It loads
`models/selection.json`, validates the entered ordered weekly history, and
returns all three predictions. Use one student-course attempt, consecutive weeks,
and only information available by the chosen cutoff.

For a compact non-interactive example, run `notebooks/03_demo.ipynb`.

Never provide GPA, final course score, course outcome, learning pace, or
`final_exam_score_audit_only` as inputs. The full input contract is in
`docs/input_contract.md`.

## 3. Reproduce the data and validation comparison

Run these commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.generate_dataset
.\.venv\Scripts\python.exe -m src.eda
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The generator uses seed `20260730`, keeps students disjoint between train,
validation, and test, and creates examples only from weeks available at each
cutoff. `src.train` compares all candidates on validation data and then freezes
the V2.3 configured mapping from `src.config.USER_SELECTED_MODELS`:

```text
gpa     -> mlp_adam
outcome -> linear
pace    -> linear
```

The frozen manifest records the seed, selection method, validation metric, and
validation rank in `models/selection.json`.

## 4. Evaluate candidate models

Inspect the validation rankings before changing the configured model mapping:

```text
reports/metrics/gpa_model_ranking.csv
reports/metrics/outcome_model_ranking.csv
reports/metrics/pace_model_ranking.csv
reports/metrics/validation_model_comparison.csv
```

Use the task-specific primary metric:

- GPA: lower RMSE is better;
- course outcome: higher `f1_enroll` is better;
- learning pace: higher macro-F1 is better.

The tables additionally provide training time, median inference time, serialized
model size, and class-specific metrics. Do not call a configured model an
automatic winner when its validation rank is lower; preserve both the choice and
the ranking in the report.

## 5. Held-out test evaluation

Run the following command exactly once and only after a new selection has been
frozen:

```powershell
.\.venv\Scripts\python.exe -m src.final_evaluation
```

It writes final metrics, predictions, diagnostic plots, package versions, and a
rerun guard. Do not retrain or replace the selection after inspecting held-out
metrics. To evaluate another configuration, create a new experiment with a new
seed and untouched test split.

The held-out reports currently stored in this repository are historical results
from the earlier automatic-selection experiment; they are not metrics for the
V2.3 MLP/linear/linear configuration.

## 6. Interpretation and responsible use

This is a synthetic-data educational capstone. Its outputs are advisory prompts
for human review, never automatic ranking, grading, enrolment, discipline, or
student-access decisions.
