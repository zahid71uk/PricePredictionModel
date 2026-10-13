import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import DATA_FILE
from app.main import create_app
from app.training import train_model


UPLOAD_TOKEN = "test-upload-token"


def _create_test_app(
    tmp_path,
    *,
    enabled=True,
    token=UPLOAD_TOKEN,
    max_bytes=5 * 1024 * 1024,
    market_analysis_api_url=None,
    cache_token=None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_path = tmp_path / "active-houses.csv"
    model_path = tmp_path / "active-model.joblib"
    shutil.copyfile(DATA_FILE, dataset_path)
    train_model(dataset_path, model_path)
    application = create_app(
        model_path,
        dataset_path,
        dataset_upload_enabled=enabled,
        dataset_upload_token=token,
        market_analysis_api_url=market_analysis_api_url,
        market_analysis_cache_invalidation_token=cache_token,
        max_dataset_upload_bytes=max_bytes,
    )
    return application, dataset_path, model_path


def _upload(client, csv_bytes, token=UPLOAD_TOKEN):
    headers = {"X-Dataset-Upload-Token": token} if token is not None else {}
    return client.post(
        "/api/v1/dataset/upload",
        headers=headers,
        files={"file": ("houses.csv", csv_bytes, "text/csv")},
    )


def test_upload_promotes_dataset_and_runtime_model(tmp_path):
    application, dataset_path, model_path = _create_test_app(tmp_path)
    changed_data = pd.read_csv(DATA_FILE)
    changed_data["price"] += 30_000
    upload = changed_data.to_csv(index=False).encode("utf-8")

    with TestClient(application) as client:
        features = {
            "square_footage": 1250,
            "bedrooms": 2,
            "bathrooms": 1,
            "year_built": 1985,
            "lot_size": 5200,
            "distance_to_city_center": 3.2,
            "school_rating": 7.1,
        }
        original_prediction = client.post("/predict", json=features).json()["predicted_price"]
        response = _upload(client, upload)

        assert response.status_code == 200
        body = response.json()
        assert body["dataset_count"] == len(changed_data)
        assert body["market_analysis_cache_refreshed"] is False
        assert "refresh is pending" in body["message"]
        assert client.get("/api/v1/dataset/houses").json()["houses"][0]["price"] == 215000
        assert client.get("/model/metrics").json()["trained_at"] == body["metrics"]["trained_at"]
        prediction = client.post("/predict", json=features)
        assert prediction.status_code == 200
        assert prediction.json()["predicted_price"] == pytest.approx(original_prediction + 30_000)

    assert pd.read_csv(dataset_path)["price"].iloc[0] == 215000
    assert model_path.is_file()


def test_upload_is_disabled_and_requires_valid_token(tmp_path):
    application, _, _ = _create_test_app(tmp_path, enabled=False)
    with TestClient(application) as client:
        assert _upload(client, b"not used").status_code == 404

    application, _, _ = _create_test_app(tmp_path / "enabled", enabled=True)
    with TestClient(application) as client:
        assert _upload(client, b"not used", token=None).status_code == 401
        assert _upload(client, b"not used", token="wrong-token").status_code == 401

    application, _, _ = _create_test_app(tmp_path / "missing-token", enabled=True, token="")
    with TestClient(application) as client:
        assert _upload(client, b"not used", token="any-token").status_code == 503


def test_invalid_or_oversized_upload_does_not_change_active_files(tmp_path):
    application, dataset_path, model_path = _create_test_app(tmp_path, max_bytes=32)
    original_dataset = dataset_path.read_bytes()
    original_model = model_path.read_bytes()

    with TestClient(application) as client:
        invalid_response = _upload(client, b"unexpected,column\n1,2\n")
        oversized_response = _upload(client, b"x" * 33)

    assert invalid_response.status_code == 422
    assert oversized_response.status_code == 413
    assert dataset_path.read_bytes() == original_dataset
    assert model_path.read_bytes() == original_model


def test_successful_promotion_invalidates_market_analysis_cache(tmp_path):
    received_tokens = []

    class InvalidationHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            received_tokens.append(self.headers.get("X-Dataset-Cache-Token"))
            self.send_response(204)
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), InvalidationHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    application, _, _ = _create_test_app(
        tmp_path,
        market_analysis_api_url=f"http://127.0.0.1:{server.server_port}",
        cache_token="cache-secret",
    )
    try:
        with TestClient(application) as client:
            response = _upload(client, pd.read_csv(DATA_FILE).to_csv(index=False).encode("utf-8"))
        assert response.status_code == 200
        assert response.json()["market_analysis_cache_refreshed"] is True
        assert received_tokens == ["cache-secret"]
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


def test_model_artifact_replace_failure_restores_active_dataset(tmp_path):
    application, dataset_path, model_path = _create_test_app(tmp_path)
    original_dataset = dataset_path.read_bytes()
    original_model = model_path.read_bytes()
    real_replace = os.replace
    replace_calls = 0

    def fail_model_replace(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated model artifact promotion failure")
        return real_replace(source, destination)

    with TestClient(application, raise_server_exceptions=False) as client:
        with patch("app.training_service.os.replace", side_effect=fail_model_replace):
            response = _upload(client, pd.read_csv(DATA_FILE).to_csv(index=False).encode("utf-8"))

    assert response.status_code == 500
    assert dataset_path.read_bytes() == original_dataset
    assert model_path.read_bytes() == original_model