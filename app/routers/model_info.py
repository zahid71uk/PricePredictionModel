from fastapi import APIRouter, Depends

from app.dependencies import get_model_service
from app.model_service import ModelService
from app.schemas import CoefficientsResponse, MetricsResponse

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/coefficients", response_model=CoefficientsResponse)
def get_coefficients(
    model_service: ModelService = Depends(get_model_service),
) -> CoefficientsResponse:
    return CoefficientsResponse(**model_service.get_coefficients())


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    model_service: ModelService = Depends(get_model_service),
) -> MetricsResponse:
    return MetricsResponse(**model_service.get_metrics())