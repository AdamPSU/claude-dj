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


@dataclass(frozen=True)
class HeadPose:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True)
class TrackContext:
    energy: float = 0.5
    cluster: str | None = None


@dataclass
class ReactionFrame:
    timestamp: float = field(default_factory=time.time)
    presence: float | None = None
    movement: float | None = None
    head_pose: HeadPose | None = None
    face: float | None = None
    raw_emotions: dict[str, float] | None = None
    emotions: dict[str, float] | None = None
    dominant_emotion: str | None = None
    emotion_confidence: float | None = None
    playback: float | None = None
    vocal: float | None = None
    source: SignalSource = SignalSource.WEBCAM


@dataclass
class Baseline:
    presence: float = 1.0
    movement: float = 0.0
    face: float = 0.0
    emotions: dict[str, float] = field(
        default_factory=lambda: {"happy": 0.0, "neutral": 1.0, "disinterested": 0.0}
    )
    captured_at: float = field(default_factory=time.time)
    frame_count: int = 0


@dataclass
class ReactionScore:
    score: float
    confidence: float
    sentiment: Sentiment
    window_start: float = 0.0
    window_end: float = 0.0
    frame_count: int = 0
    source: SignalSource = SignalSource.WEBCAM
