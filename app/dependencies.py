from fastapi import HTTPException, Request, status

from app.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    model_service = getattr(request.app.state, "model_service", None)
    if model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The prediction model is not available.",
        )
    return model_service