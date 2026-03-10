"""Audio isolator/decomposition endpoint."""

import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from app.config import JOBS_DIR
from app.utils.audio import validate_upload, save_upload
from app.services.isolator_service import run_decomposition

router = APIRouter()


@router.post("/isolator")
async def isolate_audio(
    audio: UploadFile = File(...),
    method: str = Form("nmf"),
    n_fft: int = Form(1024),
    overlap: int = Form(75),
    window: str = Form("hann"),
    max_freq: int = Form(8000),
    n_components: int = Form(5),
    threshold: float = Form(-40.0),
    peak_dist: int = Form(10),
    min_size: int = Form(30),
    morph_size: int = Form(3),
    nmf_iter: int = Form(100),
):
    """Decompose audio into components using Watershed, NMF, or CCA."""
    if method not in ("watershed", "nmf", "cca"):
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

    error = validate_upload(audio.filename, audio.size or 0)
    if error:
        raise HTTPException(status_code=400, detail=error)

    ext = Path(audio.filename).suffix
    tmp_path = await save_upload(audio, suffix=ext)

    try:
        params = {
            "n_fft": n_fft,
            "overlap": overlap,
            "window": window,
            "max_freq": max_freq,
            "n_components": n_components,
            "threshold": threshold,
            "peak_dist": peak_dist,
            "min_size": min_size,
            "morph_size": morph_size,
            "nmf_iter": nmf_iter,
        }
        job_id, result = await run_decomposition(tmp_path, method, params)
        return {
            "job_id": job_id,
            "method": method,
            **result,
        }
    finally:
        os.unlink(tmp_path)


@router.get("/isolator/results/{job_id}/{filename}")
async def get_result(job_id: str, filename: str):
    """Serve a decomposition result file (PNG or WAV)."""
    path = os.path.join(JOBS_DIR, "isolator", job_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Result not found")

    if filename.endswith(".png"):
        return FileResponse(path, media_type="image/png")
    elif filename.endswith(".wav"):
        return FileResponse(path, media_type="audio/wav")
    raise HTTPException(status_code=400, detail="Unknown file type")
