"""Wraps classifier.py model loading and inference for file-based classification."""

import sys
import os
import json
from unittest.mock import MagicMock
from functools import lru_cache

import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional
import soundfile as sf

from app.config import MODEL_DIR

# Mock sounddevice so we can import classifier.py without audio hardware
sys.modules["sounddevice"] = MagicMock()
sys.path.insert(0, "/app/tools")
from classifier import HouseholdSoundCNN, PANNsTransferModel, load_model


class ClassifierService:
    """Manages model loading and file-based inference."""

    def __init__(self, model_dir: str):
        print(f"Loading classifier model from {model_dir}...")
        self.model, self.labels, self.config = load_model(model_dir)
        self.sr = self.config["sample_rate"]
        self.duration = self.config["duration"]
        self.model_type = self.config.get("model_type", "custom_cnn")
        self.is_panns = self.model_type == "panns_cnn14"

        if not self.is_panns:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=self.config["n_fft"],
                hop_length=self.config["hop_length"],
                n_mels=self.config["n_mels"],
                f_max=self.config["f_max"],
            )
            self.amp_to_db = T.AmplitudeToDB(top_db=self.config["top_db"])

        print(f"  Model type: {self.model_type}, classes: {len(self.labels)}")

    def classify_file(self, audio_path: str, top_k: int = 5) -> list[dict]:
        """Classify an audio file and return top-k predictions."""
        import subprocess, tempfile
        # Convert non-WAV formats to WAV via ffmpeg (soundfile can't read mp3)
        load_path = audio_path
        if not audio_path.lower().endswith(".wav"):
            load_path = tempfile.mktemp(suffix=".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", "44100", "-ac", "1", load_path],
                capture_output=True, check=True,
            )
        try:
            data, sr = sf.read(load_path, dtype="float32")
        finally:
            if load_path != audio_path:
                os.unlink(load_path)
        # soundfile returns (samples,) for mono or (samples, channels) for stereo
        waveform = torch.from_numpy(data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)
        else:
            waveform = waveform.T  # (channels, samples)

        # Resample if needed
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, int(sr), self.sr)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or trim to expected duration
        n_samples = self.sr * self.duration
        if waveform.shape[1] < n_samples:
            waveform = torch.nn.functional.pad(waveform, (0, n_samples - waveform.shape[1]))
        else:
            waveform = waveform[:, :n_samples]

        with torch.no_grad():
            if self.is_panns:
                logits = self.model(waveform)
            else:
                mel = self.mel_transform(waveform)
                mel_db = self.amp_to_db(mel).unsqueeze(0)
                logits = self.model(mel_db)

            probs = torch.softmax(logits, dim=1)
            topk = torch.topk(probs, min(top_k, len(self.labels)))

        return [
            {"label": self.labels[i], "confidence": round(p.item(), 4)}
            for i, p in zip(topk.indices[0], topk.values[0])
        ]


_instance: ClassifierService | None = None


def get_classifier() -> ClassifierService:
    """Get or create the singleton classifier service."""
    global _instance
    if _instance is None:
        _instance = ClassifierService(MODEL_DIR)
    return _instance
