"""Audio classification endpoint."""

import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.utils.audio import validate_upload, save_upload
from app.services.classifier_service import get_classifier

router = APIRouter()


@router.post("/classify")
async def classify_audio(
    audio: UploadFile = File(...),
    top_k: int = Form(5),
):
    """Classify an uploaded audio file using the trained CNN model."""
    error = validate_upload(audio.filename, audio.size or 0)
    if error:
        raise HTTPException(status_code=400, detail=error)

    ext = Path(audio.filename).suffix
    tmp_path = await save_upload(audio, suffix=ext)

    try:
        svc = get_classifier()
        predictions = svc.classify_file(tmp_path, top_k=top_k)
        return {
            "predictions": predictions,
            "model_type": svc.model_type,
            "num_classes": len(svc.labels),
        }
    finally:
        os.unlink(tmp_path)
