import asyncio
import logging
import secrets
from http import HTTPStatus

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, status
from starlette.datastructures import UploadFile

from app.schemas import DatasetTrainingResponse, MetricsResponse
from app.training_service import promote_dataset

router = APIRouter(prefix="/api/v1/dataset", tags=["dataset training"])
logger = logging.getLogger(__name__)


def _authorize_upload(request: Request, supplied_token: str | None) -> None:
    if not request.app.state.dataset_upload_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    expected_token = request.app.state.dataset_upload_token
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset upload authentication is not configured.",
        )
    if supplied_token is None or not secrets.compare_digest(
        expected_token.encode("utf-8"), supplied_token.encode("utf-8")
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid upload token.")


async def _invalidate_market_analysis_cache(request: Request) -> bool:
    token = request.app.state.market_analysis_cache_invalidation_token
    if not token:
        logger.warning("Market Analysis cache invalidation is not configured")
        return False

    url = f"{request.app.state.market_analysis_api_url}/api/v1/market-analysis/internal/dataset-cache"
    headers = {"X-Dataset-Cache-Token": token}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for attempt in range(3):
            try:
                response = await client.post(url, headers=headers)
                if response.status_code == status.HTTP_204_NO_CONTENT:
                    return True
                if response.status_code < status.HTTP_500_INTERNAL_SERVER_ERROR:
                    logger.warning(
                        "Market Analysis cache invalidation attempt %d returned HTTP %d",
                        attempt + 1,
                        response.status_code,
                    )
                    break
            except httpx.HTTPError as error:
                logger.warning(
                    "Market Analysis cache invalidation attempt %d failed (%s)",
                    attempt + 1,
                    type(error).__name__,
                )
            else:
                logger.warning(
                    "Market Analysis cache invalidation attempt %d returned HTTP %d",
                    attempt + 1,
                    response.status_code,
                )
            if attempt < 2:
                await asyncio.sleep(0.2 * (attempt + 1))
    return False


@router.post("/upload", response_model=DatasetTrainingResponse)
async def upload_dataset(
    request: Request,
    x_dataset_upload_token: str | None = Header(default=None),
) -> DatasetTrainingResponse:
    _authorize_upload(request, x_dataset_upload_token)
    maximum_bytes = request.app.state.max_dataset_upload_bytes
    if maximum_bytes <= 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset upload size is not configured.",
        )
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            request_size = int(content_length)
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request size.") from error
        if request_size > maximum_bytes + 64 * 1024:
            raise HTTPException(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE, detail="CSV file is too large.")

    form = None
    try:
        form = await request.form()
        file = form.get("file")
        if not isinstance(file, UploadFile) or not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail="Choose a CSV file.")
        upload = await file.read(maximum_bytes + 1)
    finally:
        if form is not None:
            await form.close()
    if not upload:
        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail="CSV file is empty.")
    if len(upload) > maximum_bytes:
        raise HTTPException(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE, detail="CSV file is too large.")

    try:
        metrics, dataset_count = await asyncio.to_thread(promote_dataset, upload, request.app)
    except (ValueError, UnicodeDecodeError) as error:
        logger.warning("Dataset upload rejected during training validation")
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception("Dataset training or promotion failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="The dataset could not be trained and promoted.",
        ) from error

    cache_refreshed = await _invalidate_market_analysis_cache(request)
    logger.info(
        "Dataset and model promoted: rows=%d market_cache_refreshed=%s",
        dataset_count,
        cache_refreshed,
    )
    return DatasetTrainingResponse(
        dataset_count=dataset_count,
        metrics=MetricsResponse(**metrics),
        market_analysis_cache_refreshed=cache_refreshed,
        message=(
            "Dataset and model updated. Market Analysis cache refreshed."
            if cache_refreshed
            else "Dataset and model updated. Market Analysis refresh is pending."
        ),
    )