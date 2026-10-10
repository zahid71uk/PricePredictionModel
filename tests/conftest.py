import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import MODEL_ARTIFACT_PATH, TEST_DATA_FILE
from app.main import create_app
from app.training import train_model


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    if not MODEL_ARTIFACT_PATH.is_file():
        train_model()


@pytest.fixture
def client(trained_model):
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def test_houses():
    return pd.read_csv(TEST_DATA_FILE).to_dict(orient="records")