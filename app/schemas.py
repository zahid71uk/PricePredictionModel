from pydantic import BaseModel, ConfigDict, Field

from app.config import MAX_BATCH_SIZE


class HouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    square_footage: float = Field(gt=0)
    bedrooms: int = Field(ge=0)
    bathrooms: float = Field(ge=0)
    year_built: int = Field(ge=1800, le=2100)
    lot_size: float = Field(gt=0)
    distance_to_city_center: float = Field(ge=0)
    school_rating: float = Field(ge=0, le=10)


class PredictionResponse(BaseModel):
    predicted_price: float


class BatchPredictionRequest(BaseModel):
    houses: list[HouseFeatures] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    count: int


class CoefficientsResponse(BaseModel):
    intercept: float
    coefficients: dict[str, float]


class MetricsResponse(BaseModel):
    r2_score: float
    mae: float
    rmse: float
    train_samples: int
    test_samples: int
    trained_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool