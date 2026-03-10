"""Server-side monitor session — runs the full monitor.py pipeline on audio chunks
streamed from the browser's Web Audio API."""

import time
import uuid
import numpy as np
import torch

import sys
from unittest.mock import MagicMock
sys.modules["sounddevice"] = MagicMock()
sys.path.insert(0, "/app/tools")

from monitor import (
    SignalCleaner, FeatureExtractor, SoundTracker, ContextEngine,
)
from app.services.classifier_service import get_classifier


class MonitorSession:
    """One active monitoring session, tied to a browser tab."""

    def __init__(self, sensitivity: float = 0.4):
        svc = get_classifier()
        self.model = svc.model
        self.labels = svc.labels
        self.config = svc.config
        self.sr = svc.sr
        self.duration = svc.duration
        self.n_samples = self.sr * self.duration

        self.cleaner = SignalCleaner(n_components=4, n_noise=1)
        self.extractor = FeatureExtractor(self.config)
        self.tracker = SoundTracker(self.labels, confidence_threshold=sensitivity)
        self.context = ContextEngine()

        self.buffer = np.zeros(self.n_samples, dtype=np.float32)
        self.calibrated = False
        self.created_at = time.time()
        self.last_active = time.time()

    def feed_audio(self, pcm: np.ndarray):
        """Append PCM float32 mono audio to the rolling buffer."""
        # Resample not needed — browser sends at target rate
        self.buffer = np.roll(self.buffer, -len(pcm))
        self.buffer[-len(pcm):] = pcm
        self.last_active = time.time()

    def calibrate(self):
        """Learn noise profile from current buffer contents."""
        self.cleaner.calibrate(self.buffer.copy(), self.sr)
        self.calibrated = True

    def process(self) -> dict:
        """Run full pipeline on current buffer, return JSON-serialisable results."""
        timestamp = time.time()

        # 1. NMF signal cleaning
        audio = self.buffer.copy()
        cleaned = self.cleaner.clean(audio, self.sr) if self.calibrated else audio

        # 2. Feature extraction
        features = self.extractor.extract(cleaned)

        # 3. CNN classification
        with torch.no_grad():
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1)
            topk = torch.topk(probs, min(5, len(self.labels)))
        predictions = [
            (self.labels[i], round(p.item(), 4))
            for i, p in zip(topk.indices[0], topk.values[0])
        ]

        # 4. Sound tracking (state machine)
        events = self.tracker.update(predictions, timestamp)

        # 5. Context engine
        messages = []
        for event in events:
            response = self.context.evaluate(event)
            if response:
                messages.append(response)

        # Build active sounds list
        active_sounds = []
        for label, state in self.tracker.states.items():
            if state in (SoundTracker.ACTIVE, SoundTracker.DETECTED, SoundTracker.FADING):
                dur = self.tracker.durations.get(label, 0)
                active_sounds.append({
                    "label": label,
                    "state": state,
                    "duration": round(dur, 1),
                })

        return {
            "predictions": [
                {"label": l, "confidence": c} for l, c in predictions
            ],
            "active_sounds": active_sounds,
            "events": events,
            "messages": messages,
            "calibrated": self.calibrated,
        }


# ── Session store ───────────────────────────────────────────────────────────

_sessions: dict[str, MonitorSession] = {}
MAX_SESSIONS = 5
SESSION_TIMEOUT = 300  # 5 minutes of inactivity


def create_session(sensitivity: float = 0.4) -> str:
    """Create a new monitor session, return its ID."""
    _gc_sessions()
    if len(_sessions) >= MAX_SESSIONS:
        # Evict oldest
        oldest_id = min(_sessions, key=lambda k: _sessions[k].last_active)
        del _sessions[oldest_id]
    sid = uuid.uuid4().hex[:12]
    _sessions[sid] = MonitorSession(sensitivity=sensitivity)
    return sid


def get_session(sid: str) -> MonitorSession | None:
    return _sessions.get(sid)


def remove_session(sid: str):
    _sessions.pop(sid, None)


def _gc_sessions():
    """Remove sessions that have been inactive too long."""
    now = time.time()
    expired = [k for k, v in _sessions.items()
               if now - v.last_active > SESSION_TIMEOUT]
    for k in expired:
        del _sessions[k]
