# Reproducing SRG-Tracker V2.2

This guide explains what each command does so the project can be reproduced
without AI assistance.

## 1. Create the environment

Open PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project uses pandas and NumPy for data, scikit-learn for preprocessing and
baselines, XGBoost and CatBoost for boosting, PyTorch for GRU, Matplotlib/Seaborn
for reports, and ipywidgets for the notebook product.

## 2. Generate the data

```powershell
python -m src.generate_dataset
```

This creates a new deterministic split using seed `20260730`. The generator:

1. draws stable student traits and course/semester effects;
2. generates temporally dependent weekly activity;
3. creates midterm and final audit data at their allowed times;
4. constructs final score, GPA, outcome, and pace labels;
5. assigns whole students to train, validation, or test;
6. validates timing, ranges, counts, and overlap;
7. saves the quality report.

## 3. Run EDA

```powershell
python -m src.eda
```

EDA checks row and student counts, missing values, target distributions, and
student overlap. It writes tables to `reports/metrics/` and figures to
`reports/figures/`.

Missing assignment, quiz, and midterm values are expected when those assessments
have not occurred. They must not be filled from future weeks.

## 4. Run contract tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover:

- GPA boundary conversion;
- score-weight sum;
- forbidden model columns;
- history order and cutoff validation;
- midterm timing;
- required assignment and quiz;
- aggregate future-week isolation;
- explicit GRU padding mask;
- student-disjoint splits;
- saved full-history inference.

Fix a failed test before training. Do not weaken a leakage test to make it pass.

## 5. Train and select models

```powershell
python -m src.train
```

The command builds one training example per attempt at weeks 4, 7, 10, and 14.
It compares:

- frozen V1 latest-week reference;
- Dummy baseline;
- Ridge or Logistic Regression;
- HistGradientBoosting;
- XGBoost;
- CatBoost;
- MLP with Adam;
- shared multi-task GRU with Adam.

Tabular models receive deterministic history aggregates. GRU receives the raw
chronological weekly tensor. `pack_padded_sequence` ensures padded weeks are not
treated as evidence.

Only train and validation are used here. Winners are written to
`models/selection.json` before test evaluation.

## 6. Understand the rankings

Open `notebooks/02_baseline.ipynb` or inspect:

```text
reports/metrics/gpa_model_ranking.csv
reports/metrics/outcome_model_ranking.csv
reports/metrics/pace_model_ranking.csv
reports/metrics/validation_model_comparison.csv
```

GPA is ranked by ascending RMSE. Outcome is ranked by descending enroll F1. Pace
is ranked by descending macro-F1. The tables also contain serialized size,
training time, and median inference time after warm-up.

## 7. Run final evaluation once

```powershell
python -m src.final_evaluation
```

Do this only after selection is frozen. The module reads the held-out test split,
writes final metrics and diagnostics, and creates a local guard file. It refuses
to rerun automatically because repeatedly inspecting test results would turn the
test split into another validation split.

The evaluation saves:

- overall and cutoff metrics;
- row-level predictions;
- course/cutoff error breakdown;
- outcome and pace calibration;
- normalized confusion matrices;
- observed-vs-predicted GPA;
- environment versions.

## 8. Use the notebooks

Start Jupyter:

```powershell
jupyter notebook
```

Run in order:

1. `01_eda.ipynb` — data structure, automated tests, missingness, targets;
2. `02_baseline.ipynb` — validation ranking and frozen selection;
3. `03_demo.ipynb` — one synthetic week-7 history;
4. `04_final_evaluation.ipynb` — saved held-out results;
5. `final_product.ipynb` — editable interactive history form.

Notebook logic imports from `src`; changing only a notebook must not create a
different training or evaluation implementation.

## 9. Enter a custom history

Open `final_product.ipynb`, edit the JSON history, and press **Run prediction**.
Keep all weeks consecutive and use the same attempt, course, and semester.

Do not enter final score, GPA, outcome, pace, or final-exam audit data. See
`docs/input_contract.md` for examples and validation rules.

## 10. Responsible interpretation

The model learned patterns created by a synthetic generator. A high probability
does not prove that a real student will fail. The output should only demonstrate
an engineering workflow. A real deployment would require approved data access,
privacy review, representativeness and fairness evaluation, calibration on the
institution, human oversight, monitoring, and an appeal process.

V3.0 may add a dedicated frontend. Frontend work is intentionally outside V2.2.
