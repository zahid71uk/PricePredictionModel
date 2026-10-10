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