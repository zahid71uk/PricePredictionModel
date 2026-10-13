import logging
import os
import threading
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import (
    DATASET_UPLOAD_ENABLED,
    DATASET_UPLOAD_TOKEN,
    DATA_FILE,
    MARKET_ANALYSIS_API_URL,
    MARKET_ANALYSIS_CACHE_INVALIDATION_TOKEN,
    MAX_DATASET_UPLOAD_BYTES,
    MODEL_ARTIFACT_PATH,
)
from app.model_service import ModelNotFoundError, ModelService
from app.routers import dataset, health, model_info, prediction, training
from app.schemas import BatchPredictionRequest, HouseFeatures

logger = logging.getLogger(__name__)


def _describe_schema(schema: dict, definitions: dict) -> dict:
    reference = schema.get("$ref")
    if reference:
        return _describe_schema(definitions[reference.rsplit("/", 1)[-1]], definitions)

    description = {"type": schema.get("type", "object")}
    constraints = []
    for key, label in (
        ("exclusiveMinimum", "greater than"),
        ("minimum", "at least"),
        ("exclusiveMaximum", "less than"),
        ("maximum", "at most"),
        ("minItems", "at least"),
        ("maxItems", "at most"),
    ):
        if key in schema:
            count = schema[key]
            unit = (" item" if count == 1 else " items") if key.endswith("Items") else ""
            constraints.append(f"{label} {count}{unit}")
    if constraints:
        description["constraints"] = constraints

    properties = schema.get("properties")
    if properties:
        required = set(schema.get("required", []))
        description["fields"] = [
            {
                "name": name,
                **_describe_schema(field_schema, definitions),
                "required": name in required,
            }
            for name, field_schema in properties.items()
        ]
    if "items" in schema:
        description["items"] = _describe_schema(schema["items"], definitions)
    if schema.get("additionalProperties") is False:
        description["additional_fields_allowed"] = False
    return description


def _format_error_location(location: tuple) -> str:
    parts = []
    for item in location:
        if item == "body":
            continue
        if isinstance(item, int):
            parts[-1] = f"{parts[-1]}[{item}]" if parts else f"[{item}]"
        else:
            parts.append(str(item))
    return ".".join(parts) or "request body"


def create_app(
    model_path=MODEL_ARTIFACT_PATH,
    dataset_path=DATA_FILE,
    *,
    dataset_upload_enabled: bool | None = None,
    dataset_upload_token: str | None = None,
    market_analysis_api_url: str | None = None,
    market_analysis_cache_invalidation_token: str | None = None,
    max_dataset_upload_bytes: int = MAX_DATASET_UPLOAD_BYTES,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            application.state.model_service = ModelService.load(model_path)
            logger.info("Loaded model artifact from %s", model_path)
        except ModelNotFoundError:
            application.state.model_service = None
            logger.warning("Model artifact is missing; prediction endpoints are unavailable.")
        yield
        application.state.model_service = None

    application = FastAPI(
        title="House Price Prediction API",
        description="Predict housing prices from property features using linear regression.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.dataset_path = dataset_path
    application.state.model_path = model_path
    application.state.dataset_lock = threading.RLock()
    application.state.training_lock = threading.Lock()
    application.state.dataset_upload_enabled = (
        DATASET_UPLOAD_ENABLED if dataset_upload_enabled is None else dataset_upload_enabled
    )
    application.state.dataset_upload_token = (
        DATASET_UPLOAD_TOKEN if dataset_upload_token is None else dataset_upload_token
    )
    application.state.market_analysis_api_url = (
        MARKET_ANALYSIS_API_URL if market_analysis_api_url is None else market_analysis_api_url.rstrip("/")
    )
    application.state.market_analysis_cache_invalidation_token = (
        MARKET_ANALYSIS_CACHE_INVALIDATION_TOKEN
        if market_analysis_cache_invalidation_token is None
        else market_analysis_cache_invalidation_token
    )
    application.state.max_dataset_upload_bytes = max_dataset_upload_bytes

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_models = {
            "/predict": HouseFeatures,
            "/predict/batch": BatchPredictionRequest,
        }
        request_model = request_models.get(request.url.path)
        expected = None
        if request_model is not None:
            schema = request_model.model_json_schema()
            expected = _describe_schema(schema, schema.get("$defs", {}))

        return JSONResponse(
            status_code=422,
            content={
                "message": "Request data is invalid.",
                "errors": [
                    {
                        "field": _format_error_location(error["loc"]),
                        "message": error["msg"],
                    }
                    for error in exc.errors()
                ],
                "expected": expected,
            },
        )

    application.include_router(health.router)
    application.include_router(dataset.router)
    application.include_router(prediction.router)
    application.include_router(model_info.router)
    application.include_router(training.router)
    return application


app = create_app()