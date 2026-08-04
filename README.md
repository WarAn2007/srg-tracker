# SRG-Tracker 4.0 UI (web-version)

**Student:** IBRAGIMOV ANVAR

**Selected project track:** SRG Tracker - Student Ranking & GPA Tracker

**Version:** V4 local WEB release


SRG-Tracker is a local web application that gives advisory predictions from a student's observed weekly course history. It estimates final GPA, course outcome, and learning pace. The project does not rank students and must not be used for automated academic decisions.

All model choices, validation comparisons, held-out evaluations, and research artefacts are stored on branch `training-version`. This branch, `web-version`, contains the final local React UI and FastAPI adapter, desktop and mobile versions. Switch branches to review earlier versions.

## What is new in V4.0

- A typed, thread-safe model registry loads all three `.joblib` pipelines once at FastAPI startup.
- `/api/health` reports registry readiness and active model metadata.
- `/api/model-info` reports each task, model type, validation metric/value, rank, selection method, and artifact availability.
- Settings can validate and activate a trusted compatible model for one task at a time.
- Prediction history is encrypted locally with AES-GCM and can be imported/exported as a binary `.srg-history` file.
- Lightfall provides an adjustable WebGL background with four color controls, saved local preferences, and reduced-motion support.
- A four-step accessible workflow separates identity, weekly observations, validation preview, and results.
- Friendly results repeat Student ID, Course ID, Semester, cutoff week, and uncertainty context.

## 1. Problem statement

Students and instructors need an early, reproducible indication of academic progress before a course ends. SRG Tracker processes the complete ordered weekly history available through a submitted cutoff and estimates three signals: final GPA, risk of a repeat-required outcome, and learning pace. The outputs are intended to support discussion and timely help, never automatic academic decisions.

## 2. Selected project track

**SRG Tracker - Student Ranking & GPA Tracker.** The implemented V4 scope is the local UI and early-warning workflow; individual ranking is explicitly out of scope.


## 3. Dataset source

The project uses a deterministic, fully synthetic educational dataset generated with seed `20260730`. It contains 450 student-disjoint course attempts, each with weekly records for weeks 1-14 (63,000 weekly rows in total). The feature ideas and correlation exploration were inspired by the Open University Learning Analytics Dataset (OULAD), but no real OULAD learner records are included or predicted. Supporting exploratory images in `training-version` branch are [corr-matrix-oulad(1).png](reports/figures/corr-matrix-oulad(1).png) and [corr-matrix-oulad(2).png](reports/figures/corr-matrix-oulad(2).png).

| Split | Students | Attempts | Weekly rows |
| --- | ---: | ---: | ---: |
| Train | 300 | 3,000 | 42,000 |
| Validation | 100 | 1,000 | 14,000 |
| Test | 50 | 500 | 7,000 |

Following `training-version\docs\training_and_deployment.md` you can upload your own dataset, choose new model to train and use it for web, desktop and moblie versions.

## 4. ML task type

This is a multi-task supervised-learning project with separate selected models:

- Regression: predict `final_gpa` on a 0.00-4.50 scale.
- Binary classification: predict `course_outcome` (`pass` or `enroll`, where `enroll` means failed/repeat-required).
- Multiclass classification: predict `learning_pace` (`behind`, `on_track`, or `ahead`).

## Included models

The `models/` directory contains the ready-to-use model artifacts required by the application:

- Ridge Linear Regression for predicted final GPA.
- Linear classification for course outcome (`pass` or `enroll`).
- Linear classification for learning pace (`behind`, `on_track`, or `ahead`).

`enroll` means a failed or repeat-required outcome. The models are loaded at runtime; this repository is a demo application, not a training pipeline.

## Included model mapping

The packaged runtime initially uses Ridge regression for GPA and Logistic Regression pipelines for outcome and pace. `models/selection.json` records the exact mapping and validation context. A model uploaded in Settings is explicitly recorded as `user_uploaded_trusted_artifact`; it is not called an automatic winner and does not inherit an older held-out evaluation.


## 5. Pipeline and system architecture


```text
Browser (React + TypeScript + Vite)
  |-- weekly history --> FastAPI validation --> feature aggregation
  |                                             |-- cached GPA model
  |                                             |-- cached outcome model
  |                                             `-- cached pace model
  |<-- advisory result ------------------------- ModelRegistry
  |
  `-- AES-GCM encrypted history --> browser localStorage / binary export
```

Model training remains in the separate `last-change` project. V4.0 only loads approved artifacts from `models/`; it never retrains them.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Set-Location frontend
npm.cmd install
npm.cmd run build
Set-Location ..

.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. See [instructions.md](instructions.md) for setup, verification, deployment, packaging, and troubleshooting. Here you can find a way to import a new model trained on a new dataset


## Repository layout

```text
api/             FastAPI application and public schemas
frontend/        React/TypeScript UI and Lightfall component
models/          approved runtime artifacts and manifest
src/             validation, features, inference, and ModelRegistry
tests/           registry, API, leakage, and schema tests
docs/            request contract
future_update.md deferred product ideas
```


## API

- `GET /api/health` - registry readiness and active models.
- `GET /api/model-info` - validation metadata and artifact availability.
- `POST /api/predict` - leakage-safe advisory prediction.
- `POST /api/models/{task}` - validate and activate one trusted local `.joblib` artifact.
- `GET /docs` - interactive OpenAPI documentation.

## Responsible use

The included pipelines were trained on deterministic synthetic educational data. Predictions are uncertain pattern estimates, not causal conclusions. Do not use them for ranking, grading, admissions, discipline, enrolment, or automatic intervention. Use demo identifiers, not real personal data.

## Security notes

- Prediction history is encrypted in the browser with AES-256-GCM. The key is derived from the user's password with PBKDF2-SHA-256. The password is not stored and cannot be recovered.
- `.joblib` uses Python pickle internally and may execute code while loading. Upload only artifacts you created or independently trust.
- This local demo has no authentication, database, cloud service, or server-side persistence.


## 7. Final models and justification

Each task uses the validation winner for its own primary metric. This is more appropriate than forcing one algorithm to solve three different target types.

| Task | Selected model | Winner validation result | 2nd place | 3rd place |
| --- | --- | --- | --- | --- |
| Final GPA | Ridge Linear | RMSE **0.7750**; 0.0836 s; 10.4 KB | XGBoost: RMSE 0.7350; 0.66 s; 317 KB | CatBoost, aggregated history: RMSE 0.7277; 1.11 s training; 220 KB |
| Course outcome |  Logistic regression | F1 **0.8247**; 0.14 s; 10.5 KB | XGBoost: F1 0.8257; 0.69 s; 298 KB | Shared GRU, raw sequence: enroll F1 0.8259; 19.95 s; 50 KB | 
| Learning pace |  Logistic regression |  macro-F1 **0.7600**; 0.6 s; 11,6 KB | CatBoost: macro-F1 0.7688; 1.46 s; 406 KB | XGBoost, aggregated history: macro-F1 0.7692; 1.79 s; 877 KB | 

#### Reason: Evaluating matricses stored in `reports/metrics/validation_model_comparison.csv` shows that picked models are simpler, faster and light-weight comparing to complex like GradentBoost or RNN(GRU), and with a loss in a very small in accuracy and rmse in exchange

## 8. Evaluation metrics and results

Model selection used validation data only: GPA RMSE, `enroll` F1 for course outcome, and macro-F1 for learning pace. The final test split remained untouched until selection was frozen.

| Task | Final held-out test results |
| --- | --- |
| GPA | MAE 0.6088; RMSE 0.775; R² 0.693 |
| Course outcome | `enroll` recall 0.8605; `enroll` F1 0.8247; ROC-AUC 0.95 |
| Learning pace | macro-F1 0.7600; balanced accuracy 0.8012 |

Results are also measured at cutoffs 4, 7, 10, and 14. Detailed metrics, calibration, error slices, figures, and held-out predictions are in `training-version` branch: `reports/metrics/`, `reports/figures/`, and `reports/predictions/`.