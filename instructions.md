# SRG-Tracker 4.0 setup and operation

These instructions use relative paths and assume PowerShell is open in the V4.0 repository root.

## 1. Prerequisites

- Python 3.10 or newer
- Node.js LTS with npm
- the four approved files in `models/`: `selection.json`, `gpa_model.joblib`, `outcome_model.joblib`, and `pace_model.joblib`

Check the tools:

```powershell
python --version
node --version
npm.cmd --version
```

If `python` is unavailable on Windows, try `py`. Use one interpreter consistently when creating and running the environment.

## 2. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Frontend install and production build

```powershell
Set-Location frontend
npm.cmd install
npm.cmd run build
Set-Location ..
```

FastAPI serves `frontend/dist`. Rebuild after every frontend source change, restart FastAPI, and hard-refresh the browser.

## 4. Start the local application

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open:

- application: `http://127.0.0.1:8000`
- health: `http://127.0.0.1:8000/api/health`
- model information: `http://127.0.0.1:8000/api/model-info`
- API documentation: `http://127.0.0.1:8000/docs`

`/api/health` should report `status: ready` and `models_ready: true`. The registry loads the models during startup and prediction requests reuse those in-memory objects.

## 5. Development mode

Keep FastAPI running. In another PowerShell terminal:

```powershell
Set-Location frontend
npm.cmd run dev
```

Open the Vite address printed by the command, normally `http://127.0.0.1:5173`.

## 6. Deploy a model trained in `version-2-3`

1. Train and validate the model only in the separate training project.
2. Confirm that its pipeline accepts the exact aggregated columns in `src/config.py`.
3. Record its model type, validation metric, validation value, and validation rank. Do not reuse held-out results from a different model mapping.
4. Start V4.0 and open **Settings**.
5. Choose the task (`gpa`, `outcome`, or `pace`), enter the recorded metadata, and select the trusted `.joblib` file.
6. Select **Validate and activate model**.
7. Recheck `/api/health`, `/api/model-info`, and a real prediction.

V4.0 validates the required methods and probes the V4 aggregated-history contract before atomically replacing the task artifact and manifest. The other two in-memory models remain unchanged. The manifest marks the resulting mapping as user-configured and held-out evaluation as not run for that new mapping.

> Security: `.joblib` relies on pickle and can execute Python code during loading. Never load a downloaded or untrusted artifact.

## 7. Encrypted history

Open **History**, create a password of at least eight characters, and save a prediction. The browser stores only an AES-GCM encrypted envelope. **Export file** downloads a binary `.srg-history` file; **Import file** restores it after the correct password is entered.

There is no password recovery. Clearing browser storage removes the local copy, so export the encrypted file when the records must be retained.

## 8. Verification

Backend tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

TypeScript and production build:

```powershell
Set-Location frontend
npm.cmd run build
Set-Location ..
```

Manual API checks after starting FastAPI:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/model-info
```

Use the valid JSON example in `docs/input_contract.md` for `POST /api/predict`.

## 9. Packaging notes

### Windows `.exe`

A distributable Windows package must include the FastAPI application, Python runtime dependencies, `frontend/dist`, `models/`, and the manifest. Tools such as PyInstaller can package the Python entry point, but the resource paths and scikit-learn binary dependencies must be tested on a clean Windows machine. Rebuild the frontend before packaging.

### Android `.apk`

Do not place `.joblib` pipelines directly inside Android. They require Python and compatible scikit-learn dependencies. A mobile client should call a separately hosted FastAPI service over HTTPS. Authentication and cloud deployment are intentionally outside this local V4.0 scope.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| `model_error` in health | Read `registry_error`, restore the named artifact, and check the Python/scikit-learn versions. |
| Uploaded model is rejected | Confirm task type, feature contract, required `predict`/`predict_proba` methods, and trusted file origin. |
| Page is stale | Run the frontend build again, restart FastAPI, and hard-refresh port 8000. |
| History will not unlock | Check the password and file. A wrong password and damaged ciphertext are intentionally indistinguishable. |
| WebGL background is slow | Turn off **Moving background** in Settings or enable reduced motion in the operating system. |
| Port 8000 is occupied | Stop the previous process or choose another local port. |

## Data and Git safety

Do not commit `.venv`, `frontend/node_modules`, `frontend/dist`, `.srg-history`, secrets, local datasets, or files containing absolute computer paths. The repository contains only synthetic-data model artifacts. Student ranking is out of scope.
