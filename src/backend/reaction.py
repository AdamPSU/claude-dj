"""Reaction data model for ClaudeDJ mood detection.

Follows the Pulse-informed fuse → window → interpret pattern from the PRD.
Raw per-second frames are aggregated over a mid-song window into a single
engagement score. The agent interprets the score relative to track context.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Sentiment(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class SignalSource(Enum):
    CLI = "cli"
    WEBCAM = "webcam"
    PLAYBACK = "playback"


@dataclass
class ReactionFrame:
    """One ~1s sample of raw reaction signals (FR-5).

    Each component is 0.0–1.0. A value of None means that channel
    was unavailable (e.g., no webcam).
    """

    timestamp: float = field(default_factory=time.time)
    presence: float | None = None  # is the listener there?
    movement: float | None = None  # head/body motion magnitude
    face: float | None = None  # expression delta from baseline
    playback: float | None = None  # playback-derived signal (skip, pause, volume)
    vocal: float | None = None  # optional singing/humming cue
    source: SignalSource = SignalSource.WEBCAM


@dataclass
class Baseline:
    """Per-listener neutral baseline captured in the first ~3s (FR-4, P3).

    Reactions are scored as deltas from this baseline so that a naturally
    still person isn't read as disengaged.
    """

    presence: float = 1.0
    movement: float = 0.0
    face: float = 0.0
    captured_at: float = field(default_factory=time.time)
    frame_count: int = 0  # how many frames contributed


@dataclass
class ReactionScore:
    """Windowed aggregate engagement score (FR-6).

    Produced by aggregating ReactionFrames over a mid-song window.
    The agent reads this alongside track context to decide positive/neutral/negative.
    """

    score: float  # 0.0 (negative) – 1.0 (positive)
    confidence: float  # 0.0 (no data) – 1.0 (strong signal)
    sentiment: Sentiment
    window_start: float = 0.0
    window_end: float = 0.0
    frame_count: int = 0
    source: SignalSource = SignalSource.WEBCAM


@dataclass
class TrackReaction:
    """Full reaction trace for one track (FR-17).

    Stored in Redis per track played.
    """

    track_id: str
    frames: list[ReactionFrame] = field(default_factory=list)
    scores: list[ReactionScore] = field(default_factory=list)
    final_sentiment: Sentiment | None = None
    final_score: float | None = None


# --- Helpers ---


def cli_to_reaction_score(feedback: str) -> ReactionScore:
    """Convert a CLI feedback string to a ReactionScore (FR-8).

    CLI feedback is a first-class signal with high confidence.
    """
    now = time.time()
    mapping: dict[str, tuple[float, Sentiment]] = {
        "like": (0.85, Sentiment.POSITIVE),
        "dislike": (0.15, Sentiment.NEGATIVE),
        "meh": (0.5, Sentiment.NEUTRAL),
    }

    feedback_lower = feedback.strip().lower()
    if feedback_lower not in mapping:
        raise ValueError(f"Unknown feedback: {feedback!r}. Use like/dislike/meh.")

    score_val, sentiment = mapping[feedback_lower]
    return ReactionScore(
        score=score_val,
        confidence=1.0,  # explicit CLI input is high confidence
        sentiment=sentiment,
        window_start=now,
        window_end=now,
        frame_count=1,
        source=SignalSource.CLI,
    )


def capture_baseline(frames: list[ReactionFrame]) -> Baseline:
    """Build a neutral baseline from the first few frames (FR-4).

    Called during the first ~3s of a session.
    """
    if not frames:
        return Baseline()

    avg_movement = 0.0
    avg_face = 0.0
    count = 0

    for f in frames:
        if f.movement is not None:
            avg_movement += f.movement
        if f.face is not None:
            avg_face += f.face
        count += 1

    return Baseline(
        presence=1.0,
        movement=avg_movement / max(count, 1),
        face=avg_face / max(count, 1),
        captured_at=time.time(),
        frame_count=count,
    )


def aggregate_window(
    frames: list[ReactionFrame],
    baseline: Baseline,
) -> ReactionScore:
    """Aggregate frames over a window into one ReactionScore (FR-6).

    Scores are computed as deltas from baseline (P3).
    Components are fused with equal weight for now (P4).
    """
    if not frames:
        return ReactionScore(
            score=0.5,
            confidence=0.0,
            sentiment=Sentiment.NEUTRAL,
        )

    deltas: list[float] = []
    for f in frames:
        components: list[float] = []
        if f.movement is not None:
            components.append(f.movement - baseline.movement)
        if f.face is not None:
            components.append(f.face - baseline.face)
        if f.vocal is not None:
            components.append(f.vocal)  # no baseline for vocal
        if components:
            deltas.append(sum(components) / len(components))

    if not deltas:
        return ReactionScore(
            score=0.5,
            confidence=0.0,
            sentiment=Sentiment.NEUTRAL,
            window_start=frames[0].timestamp,
            window_end=frames[-1].timestamp,
            frame_count=len(frames),
        )

    raw_delta = sum(deltas) / len(deltas)
    # Map delta (-1..1) to score (0..1)
    score = max(0.0, min(1.0, 0.5 + raw_delta))

    # Confidence based on how many frames had data
    confidence = min(1.0, len(deltas) / max(len(frames), 1))

    # Classify sentiment
    if score >= 0.6:
        sentiment = Sentiment.POSITIVE
    elif score <= 0.4:
        sentiment = Sentiment.NEGATIVE
    else:
        sentiment = Sentiment.NEUTRAL

    return ReactionScore(
        score=round(score, 3),
        confidence=round(confidence, 3),
        sentiment=sentiment,
        window_start=frames[0].timestamp,
        window_end=frames[-1].timestamp,
        frame_count=len(frames),
    )
