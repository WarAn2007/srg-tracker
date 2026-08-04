"""FastAPI adapter for the SRG-Tracker V4.0 local application."""

from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import HealthResponse, ModelInfo, PredictionRequest, PredictionResponse
from src.inference import predict_history
from src.model_registry import ModelRegistry, ModelRegistryError


LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
MAX_MODEL_BYTES = 64 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    app.state.registry = registry
    app.state.registry_error = None
    try:
        registry.load()
    except ModelRegistryError as error:
        app.state.registry_error = str(error)
        LOGGER.error("Model registry startup error: %s", error)
    yield


app = FastAPI(
    title="SRG-Tracker API",
    version="4.0",
    description="Local advisory prediction from observed weekly course history.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def error_response(status: int, code: str, message: str, fields: list[dict] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "fields": fields or []}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, error: RequestValidationError) -> JSONResponse:
    fields = [
        {"path": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
        for item in error.errors()
    ]
    return error_response(422, "request_validation_failed", "Check the highlighted request fields.", fields)


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    registry: ModelRegistry = request.app.state.registry
    registry_error = request.app.state.registry_error
    return HealthResponse(
        status="ready" if registry.ready else "model_error",
        models_ready=registry.ready,
        registry_error=registry_error,
        models=[ModelInfo(**item) for item in registry.model_info()] if registry.ready else [],
    )


@app.get("/api/model-info", response_model=list[ModelInfo])
def model_info(request: Request):
    registry: ModelRegistry = request.app.state.registry
    if not registry.ready:
        return error_response(503, "model_registry_unavailable", request.app.state.registry_error or "Models are not ready.")
    return [ModelInfo(**item) for item in registry.model_info()]


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    registry: ModelRegistry = request.app.state.registry
    if not registry.ready:
        return error_response(503, "model_registry_unavailable", request.app.state.registry_error or "Models are not ready.")
    try:
        result = predict_history([row.model_dump() for row in payload.history], registry=registry)
    except (ValueError, ModelRegistryError) as error:
        return error_response(400, "prediction_input_invalid", str(error))
    return PredictionResponse(**result)


@app.post("/api/models/{task}", response_model=ModelInfo)
async def replace_model(
    task: str,
    request: Request,
    artifact: UploadFile = File(...),
    model_type: str = Form(...),
    validation_metric: str = Form(...),
    validation_value: float = Form(...),
    rank: int = Form(..., ge=1),
):
    if not artifact.filename or not artifact.filename.lower().endswith(".joblib"):
        return error_response(400, "invalid_model_file", "Select a .joblib file.")
    content = await artifact.read(MAX_MODEL_BYTES + 1)
    if len(content) > MAX_MODEL_BYTES:
        return error_response(413, "model_file_too_large", "The model file exceeds the 64 MB local limit.")
    registry: ModelRegistry = request.app.state.registry
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        info = registry.replace_model(
            task,
            temporary_path,
            model_type=model_type,
            validation_metric=validation_metric,
            validation_value=validation_value,
            rank=rank,
        )
        request.app.state.registry_error = None
        return ModelInfo(**info)
    except ModelRegistryError as error:
        return error_response(400, "incompatible_model", str(error))
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)


if FRONTEND_DIST.exists():
    assets = FRONTEND_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        candidate = FRONTEND_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
