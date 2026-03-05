"""
Record Custom Training Samples
================================
Records 5-second audio clips from the microphone for training the
sound classifier. Clips are saved as WAV files organized by class folder.

Usage:
    python record_samples.py chopping
    python record_samples.py fan
    python record_samples.py faucet

Each run records one clip, auto-numbered (e.g. chopping_001.wav).
Press Enter to start recording, Ctrl+C to quit.

Authors: Joshua Kirby & Alan Nur (with Claude Opus 4.6 LLM assistance)
Course:  TECHIN 513A — Managing Data And Signal Processing
"""

import os
import sys
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SAMPLE_RATE = 44100
DURATION = 5  # seconds, matches ESC-50 clip length
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "custom")


def get_next_filename(class_dir, class_name):
    """Find the next available numbered filename in the class directory."""
    os.makedirs(class_dir, exist_ok=True)
    existing = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
    return os.path.join(class_dir, f"{class_name}_{len(existing)+1:03d}.wav")


def record_clip():
    """Record a single 5-second clip from the default microphone."""
    print(f"  Recording {DURATION} seconds...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float64")
    sd.wait()
    # Flatten to mono 1D array
    audio = audio.flatten()
    # Normalize to [-1, 1]
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    return audio


def save_clip(audio, path):
    """Save audio as 16-bit WAV."""
    pcm = (audio * 32767).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, pcm)
    print(f"  Saved: {path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python record_samples.py <class_name>")
        print("Example classes: chopping, fan, faucet, knocking, typing")
        sys.exit(1)

    class_name = sys.argv[1].lower().strip()
    class_dir = os.path.join(DATA_DIR, class_name)
    print(f"\nRecording samples for class: '{class_name}'")
    print(f"Saving to: {class_dir}")
    print(f"Duration: {DURATION}s at {SAMPLE_RATE} Hz")
    print("Press Enter to record, Ctrl+C to quit.\n")

    count = 0
    try:
        while True:
            input(f"[{class_name} #{count+1}] Press Enter to start recording...")
            audio = record_clip()
            path = get_next_filename(class_dir, class_name)
            save_clip(audio, path)
            count += 1
            print()
    except KeyboardInterrupt:
        print(f"\nDone. Recorded {count} clip(s) for '{class_name}'.")


if __name__ == "__main__":
    main()
