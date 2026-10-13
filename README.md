# House Price Prediction API

A small FastAPI service that trains and serves a scikit-learn linear regression model using the supplied housing dataset.

## Requirements

- Python 3.12 or newer

## Setup

```### If you get error when activating virtual environment us ethe below command to bypass policy issue
```### Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

From this directory, create a virtual environment with Python 3.12 (the minimum supported version), then install the project and development dependencies:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For a newer Python version, replace `3.12` and `.venv312` with that version (for example, `3.14` and `.venv314`). Use a separate virtual environment for each Python version; do not reuse one created by another interpreter.

## Train and run

```powershell
python -m app.training
python -m uvicorn app.main:app --reload
```

The training command evaluates a deterministic 80/20 holdout split and writes `modeldata/model.joblib`. The API does not train on startup. If the artifact is absent, `/health` remains available and prediction/model-information endpoints return HTTP 503.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Docker

From this directory, build the image and start the container:

```powershell
docker build -t house-price-api .
docker run --rm -p 8000:8000 --name house-price-api house-price-api
```

The image installs the application, trains the model from the bundled training CSV during the build, and runs as a non-root user. The generated model artifact is included in the image but excluded from the Docker build context. Open `http://localhost:8000/docs` to use the API. Stop the container with `Ctrl+C`.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Service and model availability |
| GET | `/api/v1/dataset/houses` | Typed source rows for downstream analysis |
| POST | `/predict` | Predict one house price |
| POST | `/predict/batch` | Predict up to 100 houses |
| GET | `/model/coefficients` | Linear model intercept and feature coefficients |
| GET | `/model/metrics` | Holdout R², MAE, RMSE, and sample counts |

Example single prediction:

```json
{
  "square_footage": 1550,
  "bedrooms": 3,
  "bathrooms": 2,
  "year_built": 1997,
  "lot_size": 6800,
  "distance_to_city_center": 4.1,
  "school_rating": 7.6
}
```

Batch requests use `{"houses": [{...}, {...}]}` and return an ordered `predictions` list with its `count`.

The dataset endpoint returns `{"houses": [...], "count": n}` with the source CSV's `id`, seven model features, and `price`. It does not require the trained model artifact. Invalid, missing, or empty source data returns HTTP 503.

## Tests

```powershell
python -m pytest -v
```

The tests use all rows in `modeldata/test_data_for_prediction.csv` for single and batch endpoint coverage.