from fastapi import APIRouter, Request

from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=getattr(request.app.state, "model_service", None) is not None,
    )