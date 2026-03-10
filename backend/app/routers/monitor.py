"""Monitor endpoints — browser streams mic audio chunks, backend runs full pipeline."""

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse

from app.services.monitor_service import (
    create_session, get_session, remove_session,
)

import numpy as np

router = APIRouter()


@router.post("/monitor/start")
async def start_monitor(sensitivity: float = Query(0.4, ge=0.1, le=1.0)):
    """Create a new monitor session."""
    sid = create_session(sensitivity=sensitivity)
    session = get_session(sid)
    return {
        "session_id": sid,
        "sample_rate": session.sr,
        "duration": session.duration,
    }


@router.post("/monitor/chunk/{session_id}")
async def send_chunk(session_id: str, request: Request):
    """Receive a raw PCM float32 audio chunk and return current predictions.

    The browser sends the body as raw bytes (Float32Array.buffer).
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    body = await request.body()
    if len(body) == 0:
        raise HTTPException(status_code=400, detail="Empty audio chunk")

    pcm = np.frombuffer(body, dtype=np.float32)
    session.feed_audio(pcm)

    # Auto-calibrate after first chunk fills enough buffer
    if not session.calibrated:
        elapsed = session.last_active - session.created_at
        if elapsed >= session.duration:
            session.calibrate()
            return {**session.process(), "status": "calibrated"}
        return {
            "predictions": [],
            "active_sounds": [],
            "events": [],
            "messages": [],
            "calibrated": False,
            "status": "calibrating",
            "calibration_progress": min(1.0, elapsed / session.duration),
        }

    return {**session.process(), "status": "running"}


@router.delete("/monitor/stop/{session_id}")
async def stop_monitor(session_id: str):
    """End a monitor session."""
    remove_session(session_id)
    return {"status": "stopped"}
