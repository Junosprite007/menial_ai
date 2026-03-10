"""Shared audio loading and validation utilities."""

import os
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from scipy.io import wavfile

from app.config import ALLOWED_EXTENSIONS, UPLOAD_MAX_BYTES


def validate_upload(filename: str, size: int) -> str | None:
    """Return an error message if invalid, None if OK."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    if size > UPLOAD_MAX_BYTES:
        return f"File too large ({size / 1024 / 1024:.1f} MB). Max: {UPLOAD_MAX_BYTES / 1024 / 1024:.0f} MB"
    return None


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file and return (signal, sample_rate) as float64 mono."""
    try:
        fs, raw = wavfile.read(path)
    except ValueError:
        out_path = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, out_path],
            capture_output=True, check=True,
        )
        fs, raw = wavfile.read(out_path)
        os.unlink(out_path)

    if raw.ndim == 2:
        raw = raw.mean(axis=1)

    if raw.dtype == np.int16:
        sig = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.int32:
        sig = raw.astype(np.float64) / 2147483648.0
    elif raw.dtype in (np.float32, np.float64):
        sig = raw.astype(np.float64)
    else:
        sig = raw.astype(np.float64) / max(np.max(np.abs(raw)), 1)

    return sig, fs


async def save_upload(upload_file, suffix: str = ".wav") -> str:
    """Save an uploaded file to a temp path and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await upload_file.read()
    tmp.write(content)
    tmp.close()
    return tmp.name
