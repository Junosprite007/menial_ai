"""
Real-Time Household Sound Classifier
========================================
Listens to the computer microphone and classifies household sounds
using a trained CNN model on mel spectrograms.

Usage:
    python classifier.py                   # uses default models/ directory
    python classifier.py --model-dir path  # custom model directory

The model directory must contain:
    - model.pt      (trained CNN weights)
    - labels.json   (class index → name mapping)
    - config.json   (mel spectrogram parameters)

Authors: Joshua Kirby & Alan Nur (with Claude Opus 4.6 LLM assistance)
Course:  TECHIN 513A — Managing Data And Signal Processing
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import sounddevice as sd


# ── Model architecture (must match training) ─────────────────────────────────

class HouseholdSoundCNN(nn.Module):
    """4-block CNN for classifying audio feature spectrograms."""

    def __init__(self, num_classes, n_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Load model and config ────────────────────────────────────────────────────

def load_model(model_dir):
    """Load trained model, labels, and config from the model directory."""
    labels_path = os.path.join(model_dir, "labels.json")
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "model.pt")

    for p in (labels_path, config_path, model_path):
        if not os.path.exists(p):
            print(f"Error: Missing {p}")
            print("Train the model first using training/train_classifier.ipynb")
            sys.exit(1)

    with open(labels_path) as f:
        label_data = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    labels = label_data["labels"]
    num_classes = label_data["num_classes"]

    n_channels = config.get("n_channels", 1)
    model = HouseholdSoundCNN(num_classes, n_channels=n_channels)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    return model, labels, config


# ── Real-time classification ─────────────────────────────────────────────────

def run_classifier(model_dir, top_k=3, interval=1.0):
    """Run the real-time sound classifier."""
    model, labels, config = load_model(model_dir)

    sr = config["sample_rate"]
    duration = config["duration"]
    n_samples = sr * duration

    # Build mel transform matching training params
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        f_max=config["f_max"],
    )
    amp_to_db = T.AmplitudeToDB(top_db=config["top_db"])

    # Ring buffer for rolling audio window
    buffer = np.zeros(n_samples, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        buffer = np.roll(buffer, -len(mono))
        buffer[-len(mono):] = mono

    def classify():
        waveform = torch.from_numpy(buffer.copy()).unsqueeze(0)  # (1, n_samples)
        mel = mel_transform(waveform)
        mel_db = amp_to_db(mel).unsqueeze(0)  # (1, 1, n_mels, time)
        with torch.no_grad():
            logits = model(mel_db)
            probs = torch.softmax(logits, dim=1)
            topk = torch.topk(probs, min(top_k, len(labels)))
        return [(labels[i], p.item()) for i, p in
                zip(topk.indices[0], topk.values[0])]

    # ── Start listening ──
    print("=" * 50)
    print("  Household Sound Classifier")
    print("=" * 50)
    print(f"  Model:       {model_dir}")
    print(f"  Classes:     {len(labels)}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Buffer:      {duration}s rolling window")
    print(f"  Interval:    {interval}s between predictions")
    print("=" * 50)
    print("  Listening... (Ctrl+C to stop)\n")

    stream = sd.InputStream(
        callback=audio_callback,
        samplerate=sr,
        channels=1,
        blocksize=int(sr * 0.1),  # 100ms blocks
    )
    stream.start()

    # Wait for the buffer to fill before first prediction
    print(f"  Filling buffer ({duration}s)...")
    time.sleep(duration)
    print()

    try:
        while True:
            results = classify()
            # Format output
            top_label, top_conf = results[0]
            bar = "#" * int(top_conf * 30)
            print(f"  >>> {top_label:<25s} {top_conf:5.1%}  {bar}")
            for label, conf in results[1:]:
                print(f"      {label:<25s} {conf:5.1%}")
            print()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        stream.stop()
        stream.close()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time household sound classifier")
    parser.add_argument("--model-dir", default="models",
                        help="Directory containing model.pt, labels.json, config.json")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top predictions to show (default: 3)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between predictions (default: 1.0)")
    args = parser.parse_args()

    run_classifier(args.model_dir, args.top_k, args.interval)


if __name__ == "__main__":
    main()
