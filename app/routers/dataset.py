import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.config import FEATURE_COLUMNS, TARGET_COLUMN
from app.schemas import DatasetHouse, DatasetResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["dataset"])


@router.get("/api/v1/dataset/houses", response_model=DatasetResponse)
def get_dataset(request: Request) -> DatasetResponse:
    if getattr(request.app.state, "model_service", None) is None:
        return DatasetResponse(houses=[], count=0)

    data_path = request.app.state.dataset_path
    try:
        with request.app.state.dataset_lock:
            data = pd.read_csv(data_path)
        required_columns = {"id", *FEATURE_COLUMNS, TARGET_COLUMN}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns or data.empty:
            raise ValueError("Dataset is empty or missing required columns.")

        houses = [DatasetHouse.model_validate(row) for row in data.to_dict(orient="records")]
    except (OSError, ValueError, pd.errors.ParserError, ValidationError) as error:
        logger.exception("Unable to read the housing dataset from %s", data_path)
        raise HTTPException(
            status_code=503,
            detail="The housing dataset is unavailable or invalid.",
        ) from error

    return DatasetResponse(houses=houses, count=len(houses))