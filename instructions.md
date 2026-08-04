# SRG-Tracker V2.3: train and deploy models

`version-2-3` branch is the training project. It creates or imports datasets, trains
models, and exports approved artifacts to the separate `web-version` checkout in your environment, the local web
application.

## Setup

Use Python 3.12 or newer from the training-project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset options

Create the deterministic synthetic dataset:

```powershell
python -m src.generate_dataset
```

Or import private labelled data. Use
`data/templates/training_records_template.csv` for the required weekly columns,
then place your CSV in `data/incoming/` (this directory is ignored by Git):

```powershell
python -m src.data_import --input data/incoming/records.csv --tasks gpa
```

The importer creates student-disjoint train, validation, and test splits. To
import three supplied splits, pass `--train`, `--validation`, and `--test`
instead. Never commit real student records.

## Selective training

Train GPA candidates only:

```powershell
python -m src.train --tasks gpa
```

Train all configured tasks and the GRU comparison:

```powershell
python -m src.train --tasks gpa outcome pace
```

Review `reports/metrics/validation_model_comparison.csv` before approving a
model. The default V2.4 GPA configuration is MLP with Adam. To train and select
the linear Ridge GPA candidate instead, run:

```powershell
python -m src.train --tasks gpa --selected-model gpa=linear
```

The linear model is a deliberate simplicity-versus-accuracy trade-off; its
recorded validation RMSE is 0.7750.

## Deploy to the web application

Deploy only the GPA artifact, preserving web outcome and pace models:

```powershell
python -m src.deploy_models --web-project 'folder/of/web-version' --tasks gpa
```

Then rebuild/restart the web project and check `GET /api/health`. The web
application loads the packaged files from its `models/` directory.

For the complete input schema, supplied-split workflow, held-out evaluation
rules, and `.exe`/`.apk` deployment guidance, read
[docs/training_and_deployment.md](docs/training_and_deployment.md).
