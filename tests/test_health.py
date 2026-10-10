from fastapi.testclient import TestClient

from app.main import create_app


def test_health_reports_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_and_model_routes_when_artifact_is_missing(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        health_response = client.get("/health")
        coefficients_response = client.get("/model/coefficients")
        metrics_response = client.get("/model/metrics")
        prediction_response = client.post(
            "/predict",
            json={
                "square_footage": 1550,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 1997,
                "lot_size": 6800,
                "distance_to_city_center": 4.1,
                "school_rating": 7.6,
            },
        )

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok", "model_loaded": False}
    assert coefficients_response.status_code == 503
    assert metrics_response.status_code == 503
    assert prediction_response.status_code == 503