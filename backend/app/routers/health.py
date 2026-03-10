"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    from app.services.classifier_service import get_classifier
    svc = get_classifier()
    return {
        "status": "ok",
        "model_loaded": svc.model is not None,
        "model_type": svc.model_type,
        "num_classes": len(svc.labels),
    }
