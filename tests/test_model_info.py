from app.config import FEATURE_COLUMNS


def test_coefficients_include_each_feature(client):
    response = client.get("/model/coefficients")

    assert response.status_code == 200
    body = response.json()
    assert set(body["coefficients"]) == set(FEATURE_COLUMNS)
    assert isinstance(body["intercept"], float)
    assert all(isinstance(value, float) for value in body["coefficients"].values())


def test_metrics_report_holdout_performance(client):
    response = client.get("/model/metrics")

    assert response.status_code == 200
    metrics = response.json()
    assert 0 <= metrics["r2_score"] <= 1
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= metrics["mae"]
    assert metrics["train_samples"] + metrics["test_samples"] == 50
    assert metrics["trained_at"]