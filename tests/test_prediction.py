import math

import pytest


@pytest.mark.parametrize("house_index", range(10))
def test_predicts_each_test_csv_house(client, test_houses, house_index):
    response = client.post("/predict", json=test_houses[house_index])

    assert response.status_code == 200
    predicted_price = response.json()["predicted_price"]
    assert math.isfinite(predicted_price)
    assert 50_000 < predicted_price < 800_000


def test_batch_prediction_matches_test_csv_rows(client, test_houses):
    response = client.post("/predict/batch", json={"houses": test_houses})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(test_houses) == 10
    assert len(body["predictions"]) == 10
    assert all(math.isfinite(price) and price > 0 for price in body["predictions"])


def test_prediction_rejects_missing_feature(client, test_houses):
    house = dict(test_houses[0])
    house.pop("school_rating")

    response = client.post("/predict", json=house)

    assert response.status_code == 422


def test_batch_prediction_rejects_empty_list(client):
    response = client.post("/predict/batch", json={"houses": []})

    assert response.status_code == 422