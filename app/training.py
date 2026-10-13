import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
from pydantic import ValidationError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.config import (
    DATA_FILE,
    FEATURE_COLUMNS,
    MODEL_ARTIFACT_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from app.schemas import DatasetHouse

logger = logging.getLogger(__name__)


def _load_training_data(data_path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    expected_columns = ["id", *FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = set(expected_columns).difference(data.columns)
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {sorted(missing_columns)}")
    if len(data) < 6:
        raise ValueError("Training data must contain at least 6 rows.")
    if not data["id"].is_unique:
        raise ValueError("Training data must contain unique row IDs.")

    try:
        for row in data[expected_columns].to_dict(orient="records"):
            DatasetHouse.model_validate(row)
    except ValidationError as error:
        raise ValueError("Training data contains invalid rows.") from error
    return data


def train_model(
    data_path=DATA_FILE,
    artifact_path=MODEL_ARTIFACT_PATH,
) -> dict:
    """Train and persist a linear regression model and its evaluation metadata."""
    data = _load_training_data(data_path)

    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    trained_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        "r2_score": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "trained_at": trained_at,
    }
    artifact = {
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "metrics": metrics,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    logger.info("Saved model artifact to %s", artifact_path)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = train_model()
    print("Model training complete")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()