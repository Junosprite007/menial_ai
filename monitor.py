"""
Household Sound Monitor
========================
Real-time monitoring system that listens for household sounds, identifies
them using a trained CNN, tracks duration, and provides contextual voice
responses when sounds become concerning.

Pipeline:  Microphone → Signal Cleaning (NMF) → Feature Extraction
           (Mel + MFCC + ZCR + STFT) → CNN Classification → Sound
           Tracking → Context Engine → Text-to-Speech

Usage:
    python monitor.py
    python monitor.py --model-dir models --sensitivity 0.4

Authors: Joshua Kirby & Alan Nur (with Claude Opus 4.6 LLM assistance)
Course:  TECHIN 513A — Managing Data And Signal Processing
"""

import os
import sys
import json
import time
import argparse
import subprocess
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import sounddevice as sd
from scipy.signal import stft, istft
from collections import defaultdict
from datetime import datetime

from classifier import HouseholdSoundCNN


# ── NMF (from isolator.py) ───────────────────────────────────────────────────

def nmf(V, n_components, max_iter=50, tol=1e-4):
    """
    Non-Negative Matrix Factorization using multiplicative update rules.
    Decomposes V ≈ W @ H where V is a magnitude spectrogram.
    """
    n_f, n_t = V.shape
    W = np.random.rand(n_f, n_components) + 0.1
    H = np.random.rand(n_components, n_t) + 0.1
    eps = 1e-10
    prev_cost = float("inf")

    for iteration in range(max_iter):
        WtV = W.T @ V
        WtWH = W.T @ W @ H + eps
        H *= WtV / WtWH

        VHt = V @ H.T
        WHHt = W @ H @ H.T + eps
        W *= VHt / WHHt

        if iteration % 10 == 0:
            cost = np.sum((V - W @ H) ** 2)
            if abs(prev_cost - cost) / (prev_cost + eps) < tol:
                break
            prev_cost = cost

    return W, H


# ── Signal Cleaner ───────────────────────────────────────────────────────────

class SignalCleaner:
    """NMF-based signal cleaning for noise reduction."""

    def __init__(self, n_components=4, n_noise=1, n_fft=1024, hop=768):
        self.n_components = n_components
        self.n_noise = n_noise
        self.n_fft = n_fft
        self.hop = hop
        self.noise_profile = None

    def calibrate(self, silence_audio, sr):
        """Learn noise profile from ambient audio."""
        _, _, Zxx = stft(silence_audio, fs=sr, nperseg=self.n_fft,
                         noverlap=self.hop)
        self.noise_profile = np.mean(np.abs(Zxx), axis=1)

    def clean(self, audio, sr):
        """Apply NMF denoising to audio buffer."""
        _, _, Zxx = stft(audio, fs=sr, nperseg=self.n_fft, noverlap=self.hop)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        W, H = nmf(mag, self.n_components)

        if self.noise_profile is not None:
            # Drop component most correlated with noise profile
            correlations = []
            for k in range(self.n_components):
                profile = self.noise_profile[:len(W[:, k])]
                corr = np.corrcoef(W[:, k], profile)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            noise_idx = set(np.argsort(correlations)[-self.n_noise:])
            keep = [k for k in range(self.n_components) if k not in noise_idx]
        else:
            energies = np.sum(W, axis=0) * np.sum(H, axis=1)
            keep = list(np.argsort(energies)[self.n_noise:])

        mag_clean = sum(np.outer(W[:, k], H[k, :]) for k in keep)
        Zxx_clean = mag_clean * np.exp(1j * phase)
        _, cleaned = istft(Zxx_clean, fs=sr, nperseg=self.n_fft,
                           noverlap=self.hop)

        return cleaned[:len(audio)].astype(np.float32)


# ── Feature Extractor ────────────────────────────────────────────────────────

class FeatureExtractor:
    """Extract multi-channel features matching the training pipeline."""

    def __init__(self, config):
        sr = config["sample_rate"]
        self.n_fft = config["n_fft"]
        self.hop = config["hop_length"]
        self.n_mels = config["n_mels"]
        self.n_samples = sr * config["duration"]
        self.n_channels = config.get("n_channels", 1)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_fft=self.n_fft, hop_length=self.hop,
            n_mels=self.n_mels, f_max=config["f_max"])
        self.mfcc_transform = T.MFCC(
            sample_rate=sr, n_mfcc=40,
            melkwargs={"n_fft": self.n_fft, "hop_length": self.hop,
                       "n_mels": self.n_mels, "f_max": config["f_max"]})
        self.stft_transform = T.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop, power=2.0)
        self.amp_to_db = T.AmplitudeToDB(top_db=config["top_db"])

    def extract(self, audio_buffer):
        """Extract feature tensor from raw audio buffer."""
        waveform = torch.from_numpy(audio_buffer.copy()).unsqueeze(0).float()

        # Channel 0: Mel spectrogram
        mel = self.amp_to_db(self.mel_transform(waveform))
        t_frames = mel.shape[-1]

        if self.n_channels == 1:
            return mel.unsqueeze(0)  # (1, 1, n_mels, T)

        # Channel 1: MFCC (40 → resize to n_mels)
        mfcc = self.mfcc_transform(waveform)
        mfcc = F.interpolate(mfcc.unsqueeze(0), size=(self.n_mels, t_frames),
                             mode='bilinear', align_corners=False).squeeze(0)

        # Channel 2: ZCR
        zcr = self._compute_zcr(audio_buffer, t_frames)

        # Channel 3: STFT magnitude (resize to n_mels)
        stft_mag = self.amp_to_db(self.stft_transform(waveform))
        stft_resized = F.interpolate(
            stft_mag.unsqueeze(0), size=(self.n_mels, t_frames),
            mode='bilinear', align_corners=False).squeeze(0)

        features = torch.cat([mel, mfcc, zcr, stft_resized], dim=0)
        return features.unsqueeze(0)  # (1, 4, n_mels, T)

    def _compute_zcr(self, signal, t_frames):
        """Compute per-frame ZCR, broadcast to (1, n_mels, t_frames)."""
        n_frames = len(signal) // self.hop + 1
        zcr = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * self.hop
            end = min(start + self.hop, len(signal))
            frame = signal[start:end]
            if len(frame) > 1:
                zcr[i] = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / len(frame)
        zcr_range = zcr.max() - zcr.min()
        if zcr_range > 0:
            zcr = (zcr - zcr.min()) / zcr_range
        zcr_2d = np.tile(zcr, (self.n_mels, 1))
        result = torch.from_numpy(zcr_2d).float().unsqueeze(0)
        if result.shape[-1] != t_frames:
            result = F.interpolate(result.unsqueeze(0),
                                   size=(self.n_mels, t_frames),
                                   mode='bilinear',
                                   align_corners=False).squeeze(0)
        return result


# ── Sound Tracker ────────────────────────────────────────────────────────────

class SoundTracker:
    """State machine that tracks sound events and their durations."""

    IDLE = "idle"
    DETECTED = "detected"
    ACTIVE = "active"
    FADING = "fading"

    def __init__(self, labels, confidence_threshold=0.5,
                 confirm_frames=3, fade_frames=5):
        self.labels = labels
        self.threshold = confidence_threshold
        self.confirm_frames = confirm_frames
        self.fade_frames = fade_frames

        self.states = {label: self.IDLE for label in labels}
        self.counters = defaultdict(int)
        self.fade_counters = defaultdict(int)
        self.start_times = {}
        self.durations = defaultdict(float)
        self._fired_milestones = defaultdict(set)

    def update(self, predictions, timestamp):
        """Update state machine, return list of events."""
        events = []
        detected_now = {label for label, conf in predictions
                        if conf >= self.threshold}

        for label in self.labels:
            prev_state = self.states[label]

            if label in detected_now:
                self.fade_counters[label] = 0
                self.counters[label] += 1

                if prev_state == self.IDLE:
                    self.states[label] = self.DETECTED

                elif prev_state == self.DETECTED:
                    if self.counters[label] >= self.confirm_frames:
                        self.states[label] = self.ACTIVE
                        self.start_times[label] = timestamp
                        self._fired_milestones[label].clear()
                        events.append({"type": "started", "sound": label,
                                       "time": timestamp})

                elif prev_state == self.ACTIVE:
                    duration = timestamp - self.start_times[label]
                    self.durations[label] = duration
                    for milestone in [30, 60, 120, 300, 600]:
                        if (duration >= milestone and
                                milestone not in self._fired_milestones[label]):
                            self._fired_milestones[label].add(milestone)
                            events.append({"type": "duration_milestone",
                                           "sound": label, "duration": duration,
                                           "milestone": milestone,
                                           "time": timestamp})

                elif prev_state == self.FADING:
                    self.states[label] = self.ACTIVE

            else:
                self.counters[label] = 0
                self.fade_counters[label] += 1

                if prev_state == self.ACTIVE:
                    self.states[label] = self.FADING

                elif prev_state == self.FADING:
                    if self.fade_counters[label] >= self.fade_frames:
                        duration = timestamp - self.start_times.get(label,
                                                                    timestamp)
                        self.states[label] = self.IDLE
                        self.durations[label] = 0
                        events.append({"type": "stopped", "sound": label,
                                       "duration": duration, "time": timestamp})

                elif prev_state == self.DETECTED:
                    self.states[label] = self.IDLE

        return events


# ── Context Engine ───────────────────────────────────────────────────────────

class ContextEngine:
    """Generate contextual responses based on sound events."""

    RULES = {
        "chopping": {
            "started": "I hear chopping. Do you need the next recipe step?",
            "milestones": {
                60: "You've been chopping for a minute. Need a hand?",
                300: ("That's a lot of chopping! "
                      "Want me to look up a faster technique?"),
            },
            "stopped": "Sounds like the chopping is done.",
        },
        "water_tap": {
            "started": "Water is running.",
            "milestones": {
                60: "The water has been running for a minute.",
                120: ("The water has been running for 2 minutes. "
                      "Did you forget to turn it off?"),
                300: "Warning: water has been running for 5 minutes!",
            },
        },
        "boiling": {
            "started": "I hear boiling water. Would you like me to set a timer?",
            "milestones": {
                120: "Your water has been boiling for 2 minutes.",
                300: ("The pot has been boiling for 5 minutes. "
                      "Should I remind you to check it?"),
                600: ("Warning: boiling for 10 minutes. "
                      "You might want to check the stove."),
            },
            "stopped": "The boiling has stopped.",
        },
        "frying": {
            "started": "Something is frying. Shall I start a timer?",
            "milestones": {
                180: "Been frying for 3 minutes. Time to flip?",
            },
        },
        "door_knock": {
            "started": "Someone is knocking at the door.",
        },
        "smoke_detector": {
            "started": ("ALERT: Smoke detector is going off! "
                        "Please check immediately."),
        },
        "dog": {
            "started": "The dog is barking. Might need attention.",
            "milestones": {
                60: "The dog has been barking for a minute.",
            },
        },
        "crying_baby": {
            "started": "A baby is crying.",
            "milestones": {
                30: "The baby has been crying for 30 seconds.",
            },
        },
        "vacuum_cleaner": {
            "milestones": {
                600: "The vacuum has been running for 10 minutes.",
            },
        },
        "washing_machine": {
            "milestones": {
                1800: "The washing machine has been running for 30 minutes.",
            },
        },
    }

    # Map ESC-50 class names to rule keys
    CATEGORY_MAP = {
        "water_drops": "water_tap",
        "pouring_water": "water_tap",
        "toilet_flush": "water_tap",
        "clock_alarm": "smoke_detector",
        "dog": "dog",
        "crying_baby": "crying_baby",
        "door_wood_knock": "door_knock",
        "vacuum_cleaner": "vacuum_cleaner",
        "washing_machine": "washing_machine",
        # Custom classes map directly by name
        "chopping": "chopping",
        "faucet": "water_tap",
        "fan": None,
        "boiling": "boiling",
        "frying": "frying",
    }

    def evaluate(self, event):
        """Return a response string or None for a given event."""
        sound = event["sound"]
        event_type = event["type"]

        rule_key = self.CATEGORY_MAP.get(sound, sound)
        if rule_key is None or rule_key not in self.RULES:
            return None

        rules = self.RULES[rule_key]

        if event_type == "duration_milestone":
            milestone = event.get("milestone", 0)
            return rules.get("milestones", {}).get(milestone)

        return rules.get(event_type)


# ── Speaker ──────────────────────────────────────────────────────────────────

class Speaker:
    """Cross-platform text-to-speech output."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.last_spoken = {}
        self.cooldown = 30.0

    def speak(self, message):
        """Speak a message aloud (non-blocking)."""
        if not self.enabled or not message:
            return

        now = time.time()
        if message in self.last_spoken:
            if now - self.last_spoken[message] < self.cooldown:
                return
        self.last_spoken[message] = now

        if platform.system() == "Darwin":
            subprocess.Popen(["say", message],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        else:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(message)
                engine.runAndWait()
            except ImportError:
                pass


# ── Terminal Display ─────────────────────────────────────────────────────────

class Display:
    """ANSI terminal dashboard for monitoring status."""

    # Colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    DIM = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(self):
        self.recent_messages = []
        self.max_messages = 8

    def render(self, tracker, predictions, message=None):
        """Clear and redraw the terminal display."""
        print("\033[2J\033[H", end="")

        now = datetime.now().strftime("%H:%M:%S")
        print(f"{self.BOLD}{'=' * 60}")
        print(f"  HOUSEHOLD SOUND MONITOR              {now}")
        print(f"{'=' * 60}{self.RESET}")

        # Current top prediction
        if predictions:
            top_label, top_conf = predictions[0]
            if top_conf > 0.6:
                color = self.GREEN
            elif top_conf > 0.3:
                color = self.YELLOW
            else:
                color = self.DIM
            bar = "#" * int(top_conf * 30)
            print(f"\n  Detected: {color}{top_label:<25s} "
                  f"{top_conf:5.1%}{self.RESET}  {bar}")
            for label, conf in predictions[1:3]:
                print(f"            {self.DIM}{label:<25s} "
                      f"{conf:5.1%}{self.RESET}")

        # Active sounds with durations
        active = [(l, s, tracker.durations[l])
                  for l, s in tracker.states.items()
                  if s in (SoundTracker.ACTIVE, SoundTracker.DETECTED)]
        if active:
            print(f"\n  {self.CYAN}Active sounds:{self.RESET}")
            for label, state, dur in active:
                mins, secs = divmod(int(dur), 60)
                dur_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                indicator = "*" if state == SoundTracker.ACTIVE else "?"
                print(f"    {indicator} {label:<25s} {dur_str}")

        # Recent alert messages
        if message:
            self.recent_messages.append(
                f"{self.DIM}[{now}]{self.RESET} {message}")
            if len(self.recent_messages) > self.max_messages:
                self.recent_messages.pop(0)

        if self.recent_messages:
            print(f"\n  {self.YELLOW}Recent alerts:{self.RESET}")
            for msg in self.recent_messages:
                print(f"    {msg}")

        print(f"\n  {self.DIM}{'=' * 56}")
        print(f"  Press Ctrl+C to stop{self.RESET}")


# ── Main Monitor ─────────────────────────────────────────────────────────────

class HouseholdMonitor:
    """Main monitoring orchestrator — ties all pipeline stages together."""

    def __init__(self, model_dir="models", sensitivity=0.4,
                 speak=True, interval=1.0):
        self.model, self.labels, self.config = self._load_model(model_dir)
        self.sr = self.config["sample_rate"]
        self.duration = self.config["duration"]
        self.n_samples = self.sr * self.duration
        self.interval = interval

        self.cleaner = SignalCleaner(n_components=4, n_noise=1)
        self.extractor = FeatureExtractor(self.config)
        self.tracker = SoundTracker(self.labels,
                                    confidence_threshold=sensitivity)
        self.context = ContextEngine()
        self.speaker = Speaker(enabled=speak)
        self.display = Display()

        self.buffer = np.zeros(self.n_samples, dtype=np.float32)

    def _load_model(self, model_dir):
        """Load trained model, labels, and config."""
        labels_path = os.path.join(model_dir, "labels.json")
        config_path = os.path.join(model_dir, "config.json")
        model_path = os.path.join(model_dir, "model.pt")

        for p in (labels_path, config_path, model_path):
            if not os.path.exists(p):
                print(f"Error: Missing {p}")
                print("Train the model first using "
                      "training/train_classifier.ipynb")
                sys.exit(1)

        with open(labels_path) as f:
            label_data = json.load(f)
        with open(config_path) as f:
            config = json.load(f)

        labels = label_data["labels"]
        n_channels = config.get("n_channels", 1)
        model = HouseholdSoundCNN(label_data["num_classes"],
                                  n_channels=n_channels)
        model.load_state_dict(torch.load(model_path, map_location="cpu",
                                         weights_only=True))
        model.eval()
        return model, labels, config

    def _audio_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        self.buffer = np.roll(self.buffer, -len(mono))
        self.buffer[-len(mono):] = mono

    def run(self):
        """Main monitoring loop."""
        print(f"\n{'=' * 60}")
        print(f"  Household Sound Monitor")
        print(f"{'=' * 60}")
        print(f"  Model:       {len(self.labels)} classes, "
              f"{self.config.get('n_channels', 1)}-channel features")
        print(f"  Pipeline:    Mic → NMF Clean → "
              f"Mel+MFCC+ZCR+STFT → CNN → Context → TTS")
        print(f"  Interval:    {self.interval}s")
        print(f"{'=' * 60}\n")

        stream = sd.InputStream(
            callback=self._audio_callback,
            samplerate=self.sr, channels=1,
            blocksize=int(self.sr * 0.1),
        )
        stream.start()

        # Calibrate noise profile from ambient audio
        print(f"  Calibrating noise profile ({self.duration}s)...")
        time.sleep(self.duration)
        self.cleaner.calibrate(self.buffer.copy(), self.sr)
        print("  Calibrated. Starting monitor...\n")
        time.sleep(1)

        try:
            while True:
                timestamp = time.time()

                # 1. Signal Cleaning (NMF)
                audio = self.buffer.copy()
                cleaned = self.cleaner.clean(audio, self.sr)

                # 2. Feature Extraction (Mel + MFCC + ZCR + STFT)
                features = self.extractor.extract(cleaned)

                # 3. CNN Classification
                with torch.no_grad():
                    logits = self.model(features)
                    probs = torch.softmax(logits, dim=1)
                    topk = torch.topk(probs, min(5, len(self.labels)))
                predictions = [(self.labels[i], p.item())
                               for i, p in zip(topk.indices[0],
                                                topk.values[0])]

                # 4. Sound Tracking (state machine)
                events = self.tracker.update(predictions, timestamp)

                # 5. Context Engine + TTS
                message = None
                for event in events:
                    response = self.context.evaluate(event)
                    if response:
                        message = response
                        self.speaker.speak(response)

                # 6. Terminal Display
                self.display.render(self.tracker, predictions, message)

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\n  Monitoring stopped.")
        finally:
            stream.stop()
            stream.close()


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time household sound monitoring system")
    parser.add_argument("--model-dir", default="models",
                        help="Directory containing model.pt, labels.json, "
                             "config.json (default: models)")
    parser.add_argument("--sensitivity", type=float, default=0.4,
                        help="Confidence threshold for detection "
                             "(0.0-1.0, default: 0.4)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between predictions (default: 1.0)")
    parser.add_argument("--no-speak", action="store_true",
                        help="Disable text-to-speech output")
    args = parser.parse_args()

    monitor = HouseholdMonitor(
        model_dir=args.model_dir,
        sensitivity=args.sensitivity,
        speak=not args.no_speak,
        interval=args.interval,
    )
    monitor.run()


if __name__ == "__main__":
    main()
