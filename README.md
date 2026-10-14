# House Price Prediction API

A small FastAPI service that trains and serves a scikit-learn linear regression model using the supplied housing dataset.

## Requirements

- Python 3.12 or newer

## Setup

If PowerShell blocks virtual environment activation, allow it for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

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

## Dataset upload

`POST /api/v1/dataset/upload` accepts a CSV with `id`, the seven model features, and `price`. It validates and trains a candidate before promoting the active CSV and model. Upload is disabled by default; enabling it requires an upload token. The legacy `python -m app.training` command remains available for manual training.

To enable uploads, configure `DATASET_UPLOAD_ENABLED=true` and `DATASET_UPLOAD_TOKEN` in both this service and the Next.js server environment. Configure the same `MARKET_ANALYSIS_CACHE_INVALIDATION_TOKEN` here and `DATASET_CACHE_INVALIDATION_TOKEN` in Market Analysis so a successful upload immediately clears its cached rows. `MAX_DATASET_UPLOAD_BYTES` defaults to `5242880` (5 MiB). Keep the services on a private network and do not commit tokens.

If Market Analysis cannot be reached after promotion, the upload response reports that refresh is pending; its configured cache TTL is the recovery fallback.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Docker

From this directory, build the image and create the shared network used by the other module containers:

```powershell
docker build -t house-price-api .
docker network create property-estimator
docker run -d --name price-prediction --network property-estimator -p 8000:8000 house-price-api
```

The image installs the application, trains the model from the bundled training CSV during the build, and runs as a non-root user. The generated model artifact is included in the image but excluded from the Docker build context. Open `http://localhost:8000/docs` to use the API. Stop it with `docker stop price-prediction` and remove it with `docker rm price-prediction`.

To test CSV uploads, pass `-e DATASET_UPLOAD_ENABLED=true -e DATASET_UPLOAD_TOKEN=$env:DATASET_UPLOAD_TOKEN` to this `docker run` command after setting a strong token in PowerShell. Use the same token in the frontend container. For immediate Market Analysis cache refresh after uploads, also pass `-e MARKET_ANALYSIS_CACHE_INVALIDATION_TOKEN=$env:DATASET_CACHE_INVALIDATION_TOKEN` and configure the same value in the Market Analysis container. Upload remains disabled when these settings are omitted.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Service and model availability |
| GET | `/api/v1/dataset/houses` | Typed source rows for downstream analysis |
| POST | `/api/v1/dataset/upload` | Validate, train, and promote a CSV (disabled by default; requires `X-Dataset-Upload-Token`) |
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

The dataset endpoint returns `{"houses": [...], "count": n}` with the source CSV's `id`, seven model features, and `price` when a model is active. If no model is loaded, it returns an empty dataset so downstream analysis does not display stale source rows. Invalid, missing, or empty source data returns HTTP 503 when a model is active.

## Tests

```powershell
python -m pytest -v
```

The tests use all rows in `modeldata/test_data_for_prediction.csv` for single and batch endpoint coverage.