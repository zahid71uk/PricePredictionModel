import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("APP_ROOT", Path(__file__).resolve().parent.parent)
).resolve()
MODEL_DATA_DIR = PROJECT_ROOT / "modeldata"
DATA_FILE = MODEL_DATA_DIR / "house_price_dataset.csv"
TEST_DATA_FILE = MODEL_DATA_DIR / "test_data_for_prediction.csv"
MODEL_ARTIFACT_PATH = MODEL_DATA_DIR / "model.joblib"

FEATURE_COLUMNS = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating",
]
TARGET_COLUMN = "price"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_BATCH_SIZE = 100
MAX_DATASET_UPLOAD_BYTES = int(os.environ.get("MAX_DATASET_UPLOAD_BYTES", 5 * 1024 * 1024))
DATASET_UPLOAD_ENABLED = os.environ.get("DATASET_UPLOAD_ENABLED", "false").lower() == "true"
DATASET_UPLOAD_TOKEN = os.environ.get("DATASET_UPLOAD_TOKEN", "")
MARKET_ANALYSIS_API_URL = os.environ.get(
    "MARKET_ANALYSIS_API_URL", "http://127.0.0.1:8082"
).rstrip("/")
MARKET_ANALYSIS_CACHE_INVALIDATION_TOKEN = os.environ.get(
    "MARKET_ANALYSIS_CACHE_INVALIDATION_TOKEN", ""
)