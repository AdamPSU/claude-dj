"""ClaudeDJ Backend API.

Exposes reaction pipeline endpoints for the MCP server and agent.
The Reactor runs in the background, fusing webcam + CLI signals.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from reactor import Reactor

app = FastAPI(title="ClaudeDJ Backend", version="0.1.0")

# Reactor instance — webcam enabled by default, disable with CLAUDEDJ_NO_WEBCAM=1
_enable_webcam = os.environ.get("CLAUDEDJ_NO_WEBCAM", "0") != "1"
reactor = Reactor(enable_webcam=_enable_webcam)


@app.on_event("startup")
def startup():
    reactor.start()


@app.on_event("shutdown")
def shutdown():
    reactor.stop()


@app.get("/")
def read_root() -> dict[str, str]:
    return {"service": "claude-dj-backend", "status": "ok"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# --- Reaction endpoints ---


@app.get("/reaction/score")
def get_reaction_score(window_seconds: float = 15.0) -> dict:
    """Get the current engagement score over the last N seconds."""
    score = reactor.get_current_score(window_seconds=window_seconds)
    return {
        "score": score.score,
        "confidence": score.confidence,
        "sentiment": score.sentiment.value,
        "source": score.source.value,
        "frame_count": score.frame_count,
    }


class FeedbackRequest(BaseModel):
    feedback: str  # "like", "dislike", or "meh"


@app.post("/reaction/feedback")
def post_feedback(req: FeedbackRequest) -> dict:
    """Submit CLI feedback (like/dislike/meh)."""
    try:
        score = reactor.add_cli_feedback(req.feedback)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "score": score.score,
        "confidence": score.confidence,
        "sentiment": score.sentiment.value,
    }


@app.get("/reaction/summary")
def get_reaction_summary() -> dict:
    """Get compact reaction summary for agent decision bundle (FR-20)."""
    return reactor.get_summary()


@app.get("/reaction/trend")
def get_reaction_trend(windows: int = 3, window_seconds: float = 10.0) -> dict:
    """Get engagement trend over recent windows."""
    trend = reactor.get_trend(windows=windows, window_seconds=window_seconds)
    return {
        "scores": [
            {
                "score": s.score,
                "confidence": s.confidence,
                "sentiment": s.sentiment.value,
            }
            for s in trend
        ],
    }


class TrackContextRequest(BaseModel):
    energy: float = 0.5
    valence: float = 0.5
    cluster: str | None = None


@app.post("/reaction/track-context")
def post_track_context(req: TrackContextRequest) -> dict:
    """Set current track context for context-conditioned scoring (FR-7)."""
    reactor.set_track_context(energy=req.energy, valence=req.valence, cluster=req.cluster)
    return {"energy": req.energy, "valence": req.valence, "cluster": req.cluster}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
