import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.config import MODEL_ARTIFACT_PATH
from app.model_service import ModelNotFoundError, ModelService
from app.routers import health, model_info, prediction

logger = logging.getLogger(__name__)


def create_app(model_path=MODEL_ARTIFACT_PATH) -> FastAPI:
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
    application.include_router(health.router)
    application.include_router(prediction.router)
    application.include_router(model_info.router)
    return application


app = create_app()