"""
Flask server that bridges the HTML Fourier Explorer with the Python Isolator.

Usage:
    python server.py
    Then open http://localhost:5050 in your browser.

Authors: Joshua Kirby & Alan Nur (with Claude Opus 4.6 LLM assistance)
Course:  TECHIN 513A — Managing Data And Signal Processing
"""

import os
import tempfile
import subprocess
import threading
from flask import Flask, request, jsonify, send_from_directory

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(PROJECT_DIR, "static")
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(PROJECT_DIR, "fourier_explorer.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify(error="No audio file provided"), 400

    # Validate file extension
    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1].lower() or ".wav"
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify(error=f"Unsupported file type: {suffix}"), 400

    # Validate file size
    audio_file.seek(0, os.SEEK_END)
    size = audio_file.tell()
    audio_file.seek(0)
    if size > MAX_UPLOAD_BYTES:
        return jsonify(error="File too large (max 50 MB)"), 400

    # Save uploaded audio to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=PROJECT_DIR)
    audio_file.save(tmp)
    tmp.close()

    # Read STFT parameters from the form
    nfft = request.form.get("nfft", "1024")
    overlap = request.form.get("overlap", "75")
    window = request.form.get("window", "hann")
    max_freq = request.form.get("maxFreq", "8000")

    # Launch the isolator in a separate thread so the server responds immediately
    def run_isolator():
        cmd = [
            "python", os.path.join(PROJECT_DIR, "isolator.py"),
            tmp.name,
            "--nfft", nfft,
            "--overlap", overlap,
            "--window", window,
            "--maxfreq", max_freq,
        ]
        try:
            subprocess.run(cmd)
        finally:
            # Clean up temp file even if isolator crashes
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    threading.Thread(target=run_isolator, daemon=True).start()

    return jsonify(status="ok", message="Isolator launched")


if __name__ == "__main__":
    print("=" * 50)
    print("  Fourier Explorer + Python Isolator Server")
    print("  Open http://localhost:5050 in your browser")
    print("=" * 50)
    app.run(debug=False, port=5050, use_reloader=False)
