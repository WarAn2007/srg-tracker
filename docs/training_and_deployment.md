# Training and deployment guide

`version-2-3` is the training project. `web-version` is the web
application branch that loads packaged `.joblib` models. Keep them as separate
projects: train and evaluate here, then explicitly deploy only approved model
artifacts to the web project.

These instructions assume two sibling checkouts: `version-2-3` for training and `web-version` for the web application.


## 1. Install the training environment

From the root of this project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Create a synthetic dataset

Use this when you need a reproducible demonstration dataset:

```powershell
python -m src.generate_dataset
```

It writes `train.csv`, `validation.csv`, and `test.csv` to `data/processed/`.
Do not use a final test result to choose a model.

## 3. Import your own labelled dataset

Each row represents one observed week. Use
`data/templates/training_records_template.csv` as the column template. For a
task, the dataset must include its target column:

- `gpa` requires `final_gpa` from 0.0 to 4.5.
- `outcome` requires `course_outcome`: `pass` or `enroll`.
- `pace` requires `learning_pace`: `behind`, `on_track`, or `ahead`.

All data must also include `student_id`, `attempt_id`, course identifiers, and
the observed weekly fields in the template. Keep private CSV files in
`data/incoming/`; they are ignored by Git. Do not commit real student data.

### One full CSV

The importer creates deterministic 70/15/15 train, validation, and test splits
by `student_id`, so one student can never occur in more than one split:

```powershell
python -m src.data_import --input data/incoming/records.csv --tasks gpa
```

Use `--tasks gpa outcome pace` when the CSV contains all three target columns.

### Three supplied split CSV files

Use this only when the splits were created before import. The importer rejects
student overlap:

```powershell
python -m src.data_import `
  --train data/incoming/train.csv `
  --validation data/incoming/validation.csv `
  --test data/incoming/test.csv `
  --tasks gpa
```

The validated files replace `data/processed/train.csv`, `validation.csv`, and
`test.csv`. The importer writes `import_metadata.json` with row and student
counts, but never stores the source computer path.

## 4. Train selected tasks

Train only GPA candidates and save the configured MLP GPA artifact:

```powershell
python -m src.train --tasks gpa
```

Train and select the linear Ridge GPA candidate instead of the default MLP:

```powershell
python -m src.train --tasks gpa --selected-model gpa=linear
```

Train all three configured models and the research GRU comparison:

```powershell
python -m src.train --tasks gpa outcome pace
```

Training uses train data for fitting and validation data for selection. Selected
artifacts are written to `models/`, including `gpa_model.joblib` and
`selection.json`. Inspect `reports/metrics/validation_model_comparison.csv`
before approving a new artifact.

Run the held-out evaluation only after selection is frozen:

```powershell
python -m src.final_evaluation
```

For a new experiment after final evaluation, use a new dataset directory or new
experiment checkout. Do not repeatedly tune models against the held-out test.

## 5. Deploy models to the web project

The `web-version` checkout requires `gpa_model.joblib`, `outcome_model.joblib`,
`pace_model.joblib`, and `selection.json` in its `models/` directory.

To replace only the GPA model while retaining the existing outcome and pace
models in the web application, run from this training project:

```powershell
python -m src.deploy_models --web-project ../web-version --tasks gpa
```

To replace all three models:

```powershell
python -m src.deploy_models --web-project ../web-version --tasks gpa outcome pace
```

The deploy command copies only the requested artifacts and merges only those
tasks into the web project's selection manifest. Restart the API after deploy.

## 6. Run the web application

In the web project, install Python and frontend packages, build the frontend,
then start FastAPI:

```powershell
python -m pip install -r requirements.txt
Set-Location frontend
npm.cmd install
npm.cmd run build
Set-Location ..
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and confirm `/api/health` reports ready models.

## 7. Desktop and mobile clients

The `.joblib` models require Python/scikit-learn inference. For an `.exe`,
package the FastAPI application, `frontend/dist`, `models/`, and Python
dependencies together; PyInstaller is one possible packaging tool.

For an Android `.apk`, keep model inference in a FastAPI service and let the
mobile app call `POST /api/predict` over HTTPS. Running `.joblib` directly in
an APK is not supported by this project; it would require a separate model
conversion and Android inference implementation.

## Data safety

Do not send or commit real student identities, final outcomes, or sensitive
records without institutional approval. At prediction time, never include
future results or `final_exam_score_audit_only` in the input history.
