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
    body = response.json()
    assert body["message"] == "Request data is invalid."
    assert body["errors"] == [
        {"field": "school_rating", "message": "Field required"}
    ]
    expected_fields = {field["name"]: field for field in body["expected"]["fields"]}
    assert expected_fields["square_footage"]["type"] == "number"
    assert expected_fields["square_footage"]["constraints"] == ["greater than 0"]
    assert expected_fields["bedrooms"]["type"] == "integer"


def test_prediction_rejects_invalid_field_type(client, test_houses):
    house = dict(test_houses[0])
    house["bedrooms"] = "many"

    response = client.post("/predict", json=house)

    assert response.status_code == 422
    error = response.json()["errors"][0]
    assert error["field"] == "bedrooms"
    assert "integer" in error["message"]


def test_batch_prediction_rejects_empty_list(client):
    response = client.post("/predict/batch", json={"houses": []})

    assert response.status_code == 422
    body = response.json()
    assert body["errors"][0]["field"] == "houses"
    houses_field = next(
        field for field in body["expected"]["fields"] if field["name"] == "houses"
    )
    assert houses_field["type"] == "array"
    assert houses_field["constraints"] == ["at least 1 item", "at most 100 items"]


def test_batch_prediction_reports_nested_invalid_field(client, test_houses):
    house = dict(test_houses[0])
    house["bedrooms"] = "many"

    response = client.post("/predict/batch", json={"houses": [house]})

    assert response.status_code == 422
    assert response.json()["errors"][0]["field"] == "houses[0].bedrooms"