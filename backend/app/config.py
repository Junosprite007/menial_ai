"""Application configuration from environment variables."""

import os

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models/trial2/trained_models")
UPLOAD_MAX_MB = int(os.environ.get("UPLOAD_MAX_MB", "50"))
UPLOAD_MAX_BYTES = UPLOAD_MAX_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
JOBS_DIR = "/tmp/menial_jobs"
