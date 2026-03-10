"""Runs fourier_fundamentals.py as a subprocess to generate educational figures."""

import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from app.config import JOBS_DIR

# Docker copies scripts to /app/tools/; locally they live at the project root.
_DOCKER_PATH = "/app/tools/fourier_fundamentals.py"
_LOCAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "fourier_fundamentals.py"
)
FOURIER_SCRIPT = (
    _DOCKER_PATH if os.path.isfile(_DOCKER_PATH) else os.path.abspath(_LOCAL_PATH)
)


async def generate_figures(audio_path: str | None = None) -> tuple[str, list[str]]:
    """
    Run fourier_fundamentals.py and return (job_id, list_of_png_filenames).

    The script saves 5 PNG figures to its working directory.
    We run it in a job-specific output folder to collect them.
    """
    job_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join(JOBS_DIR, "fourier", job_id)
    os.makedirs(output_dir, exist_ok=True)

    cmd = ["python", FOURIER_SCRIPT]
    wav_tmp = None
    if audio_path:
        # fourier_fundamentals.py only reads WAV; convert other formats via ffmpeg
        if not audio_path.lower().endswith(".wav"):
            wav_tmp = tempfile.mktemp(suffix=".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, wav_tmp],
                capture_output=True,
                check=True,
            )
            cmd.append(wav_tmp)
        else:
            cmd.append(audio_path)

    result = subprocess.run(
        cmd,
        cwd=output_dir,
        timeout=120,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    if result.returncode != 0:
        if wav_tmp and os.path.exists(wav_tmp):
            os.unlink(wav_tmp)
        raise RuntimeError(f"Fourier script failed: {result.stderr}")

    if wav_tmp and os.path.exists(wav_tmp):
        os.unlink(wav_tmp)

    figures = sorted(Path(output_dir).glob("*.png"))
    return job_id, [f.name for f in figures]
