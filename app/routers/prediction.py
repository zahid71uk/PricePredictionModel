from fastapi import APIRouter, Depends

from app.dependencies import get_model_service
from app.model_service import ModelService
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HouseFeatures,
    PredictionResponse,
)

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict_price(
    house: HouseFeatures,
    model_service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    return PredictionResponse(predicted_price=model_service.predict_one(house))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_prices(
    request: BatchPredictionRequest,
    model_service: ModelService = Depends(get_model_service),
) -> BatchPredictionResponse:
    predictions = model_service.predict_batch(request.houses)
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))