import pandas as pd
from fastapi.testclient import TestClient

from app.main import create_app


def test_dataset_endpoint_returns_typed_rows(client):
    response = client.get("/api/v1/dataset/houses")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 50
    assert len(body["houses"]) == 50
    first_house = body["houses"][0]
    assert first_house["id"] == 1
    assert first_house["price"] == 185000
    assert first_house["square_footage"] == 1250


def test_dataset_endpoint_rejects_missing_columns(tmp_path):
    invalid_dataset = tmp_path / "invalid.csv"
    pd.DataFrame([{"id": 1, "price": 100_000}]).to_csv(invalid_dataset, index=False)

    with TestClient(create_app(dataset_path=invalid_dataset)) as client:
        response = client.get("/api/v1/dataset/houses")

    assert response.status_code == 503
    assert response.json()["detail"] == "The housing dataset is unavailable or invalid."


def test_dataset_endpoint_returns_unavailable_when_file_is_missing(tmp_path):
    missing_dataset = tmp_path / "missing.csv"

    with TestClient(create_app(dataset_path=missing_dataset)) as client:
        response = client.get("/api/v1/dataset/houses")

    assert response.status_code == 503