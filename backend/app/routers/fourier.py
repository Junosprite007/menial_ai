"""Fourier Fundamentals figure generation endpoint."""

import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.config import JOBS_DIR
from app.utils.audio import validate_upload, save_upload
from app.services.fourier_service import generate_figures

router = APIRouter()


@router.post("/fourier-fundamentals")
async def fourier_fundamentals(audio: UploadFile | None = File(None)):
    """Generate 5 educational Fourier analysis figures."""
    tmp_path = None

    if audio and audio.filename:
        error = validate_upload(audio.filename, audio.size or 0)
        if error:
            raise HTTPException(status_code=400, detail=error)
        ext = Path(audio.filename).suffix
        tmp_path = await save_upload(audio, suffix=ext)

    try:
        job_id, figure_names = await generate_figures(tmp_path)
        return {
            "job_id": job_id,
            "figures": [
                {
                    "name": name.replace(".png", ""),
                    "url": f"/api/fourier-fundamentals/figures/{job_id}/{name}",
                }
                for name in figure_names
            ],
            "source": audio.filename if audio and audio.filename else "synthetic demo",
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/fourier-fundamentals/figures/{job_id}/{filename}")
async def get_figure(job_id: str, filename: str):
    """Serve a generated figure PNG."""
    path = os.path.join(JOBS_DIR, "fourier", job_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Figure not found")
    return FileResponse(path, media_type="image/png")
