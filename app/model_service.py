from pathlib import Path

import joblib
import pandas as pd

from app.config import FEATURE_COLUMNS
from app.schemas import HouseFeatures


class ModelNotFoundError(RuntimeError):
    """Raised when the trained model artifact is unavailable."""


class ModelService:
    def __init__(self, artifact: dict) -> None:
        self._model = artifact["model"]
        self._feature_names = list(artifact["feature_names"])
        self._metrics = artifact["metrics"]
        if self._feature_names != FEATURE_COLUMNS:
            raise ValueError("Model artifact feature order does not match the API contract.")

    @classmethod
    def load(cls, path: Path) -> "ModelService":
        if not path.is_file():
            raise ModelNotFoundError(f"Model artifact not found: {path}")
        return cls(joblib.load(path))

    def predict_one(self, house: HouseFeatures) -> float:
        return self.predict_batch([house])[0]

    def predict_batch(self, houses: list[HouseFeatures]) -> list[float]:
        rows = [house.model_dump() for house in houses]
        features = pd.DataFrame(rows, columns=self._feature_names)
        return [float(value) for value in self._model.predict(features)]

    def get_coefficients(self) -> dict:
        return {
            "intercept": float(self._model.intercept_),
            "coefficients": {
                name: float(value)
                for name, value in zip(self._feature_names, self._model.coef_)
            },
        }

    def get_metrics(self) -> dict:
        return dict(self._metrics)