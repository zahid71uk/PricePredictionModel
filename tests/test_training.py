import pandas as pd
import pytest

from app.config import DATA_FILE
from app.training import train_model


def test_train_model_validates_csv_and_writes_artifact(tmp_path):
    data_path = tmp_path / "houses.csv"
    artifact_path = tmp_path / "model.joblib"
    pd.read_csv(DATA_FILE).to_csv(data_path, index=False)

    metrics = train_model(data_path, artifact_path)

    assert artifact_path.is_file()
    assert metrics["train_samples"] + metrics["test_samples"] == 50


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.drop(columns=["id"]), "missing columns"),
        (lambda data: data.head(5), "at least 6 rows"),
        (lambda data: data.assign(school_rating=11), "invalid rows"),
        (lambda data: data.assign(id=1), "unique row IDs"),
    ],
)
def test_train_model_rejects_invalid_csv_without_artifact(tmp_path, mutation, message):
    data_path = tmp_path / "invalid-houses.csv"
    artifact_path = tmp_path / "model.joblib"
    mutation(pd.read_csv(DATA_FILE)).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match=message):
        train_model(data_path, artifact_path)

    assert not artifact_path.exists()