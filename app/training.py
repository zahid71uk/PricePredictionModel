import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
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

logger = logging.getLogger(__name__)


def train_model(
    data_path=DATA_FILE,
    artifact_path=MODEL_ARTIFACT_PATH,
) -> dict:
    """Train and persist a linear regression model and its evaluation metadata."""
    data = pd.read_csv(data_path)
    expected_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = expected_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("Training data must contain at least one row.")

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