"""Menial AI — FastAPI backend serving audio analysis tools."""

import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import JOBS_DIR
from app.routers import health, classify, fourier, isolator, monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    # Pre-load the classifier model on startup
    from app.services.classifier_service import get_classifier
    get_classifier()
    yield


app = FastAPI(title="Menial AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the existing Fourier Explorer static assets at both paths:
# /explorer/static/ for when accessed via /explorer/ prefix
# /static/ for the HTML's original absolute references (/static/css/..., /static/js/...)
explorer_static = "/app/explorer/static"
if os.path.isdir(explorer_static):
    app.mount("/explorer/static", StaticFiles(directory=explorer_static), name="explorer-static")
    app.mount("/static", StaticFiles(directory=explorer_static), name="legacy-static")

# Routers
app.include_router(health.router, prefix="/api")
app.include_router(classify.router, prefix="/api")
app.include_router(fourier.router, prefix="/api")
app.include_router(isolator.router, prefix="/api")
app.include_router(monitor.router, prefix="/api")


@app.get("/explorer/")
async def explorer_index():
    """Serve the existing Fourier Explorer HTML."""
    from fastapi.responses import FileResponse
    html_path = "/app/explorer/fourier_explorer.html"
    if os.path.isfile(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {"error": "Fourier Explorer not found"}


@app.post("/analyze")
async def analyze_for_explorer(request: Request):
    """Handle the Fourier Explorer's 'Analyze in Python' button.

    Runs the isolator service headlessly and returns component results.
    """
    from app.services.isolator_service import run_decomposition

    form = await request.form()
    audio = form.get("audio")
    if not audio:
        return {"error": "No audio file provided"}

    suffix = os.path.splitext(getattr(audio, "filename", "audio.wav"))[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await audio.read()
    tmp.write(content)
    tmp.close()

    params = {
        "n_fft": int(form.get("nfft", 1024)),
        "overlap": int(form.get("overlap", 75)),
        "window": str(form.get("window", "hann")),
        "max_freq": int(form.get("maxFreq", 8000)),
        "n_components": 5,
        "threshold": -40.0,
        "peak_dist": 10,
        "min_size": 30,
        "morph_size": 3,
        "nmf_iter": 100,
    }

    try:
        job_id, result = await run_decomposition(tmp.name, "nmf", params)
        return {"status": "ok", "message": "Analysis complete", "job_id": job_id, **result}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
