from __future__ import annotations

import time

from .models import Baseline, ReactionFrame, ReactionScore, Sentiment, SignalSource, TrackContext


EMOTION_WEIGHTS: dict[str, float] = {
    "happy": 1.0,
    "neutral": 0.0,
    "disinterested": -0.5,
}

RAW_TO_COLLAPSED: dict[str, str] = {
    "happy": "happy",
    "surprise": "happy",
    "neutral": "neutral",
    "sad": "disinterested",
    "angry": "disinterested",
    "fear": "disinterested",
    "disgust": "disinterested",
}

COLLAPSED_KEYS = ("happy", "neutral", "disinterested")


def emotion_confidence(emotions: dict[str, float]) -> float:
    probs = [probability for probability in emotions.values() if probability > 0]
    if not probs:
        return 0.0
    uniform = 1.0 / max(len(emotions), len(COLLAPSED_KEYS))
    max_probability = max(probs)
    return round(max(0.0, min(1.0, (max_probability - uniform) / (1.0 - uniform))), 3)


def capture_baseline(frames: list[ReactionFrame]) -> Baseline:
    if not frames:
        return Baseline()

    movement_total = 0.0
    face_total = 0.0
    emotion_totals: dict[str, float] = {}
    movement_count = 0
    face_count = 0
    emotion_count = 0
    for frame in frames:
        if frame.movement is not None:
            movement_total += frame.movement
            movement_count += 1
        if frame.face is not None:
            face_total += frame.face
            face_count += 1
        if frame.emotions is not None:
            for key, value in frame.emotions.items():
                emotion_totals[key] = emotion_totals.get(key, 0.0) + value
            emotion_count += 1

    return Baseline(
        movement=movement_total / max(movement_count, 1),
        face=face_total / max(face_count, 1),
        emotions={key: value / max(emotion_count, 1) for key, value in emotion_totals.items()}
        if emotion_totals
        else Baseline().emotions,
        captured_at=time.time(),
        frame_count=len(frames),
    )


def aggregate_window(
    frames: list[ReactionFrame],
    baseline: Baseline,
    track_context: TrackContext | None = None,
) -> ReactionScore:
    if not frames:
        return ReactionScore(score=0.5, confidence=0.0, sentiment=Sentiment.NEUTRAL)

    movement_weight = 1.0
    face_weight = 1.0
    emotion_weight = 1.0
    vocal_weight = 1.0
    if track_context is not None:
        movement_weight = 0.5 + track_context.energy
        face_weight = 1.5 - (0.5 * track_context.energy)

    deltas: list[float] = []
    for frame in frames:
        components: list[tuple[float, float]] = []
        if frame.movement is not None:
            components.append((frame.movement - baseline.movement, movement_weight))
        if frame.face is not None:
            components.append((frame.face - baseline.face, face_weight))
        if frame.emotions is not None:
            emotion_delta = sum(
                (frame.emotions.get(key, 0.0) - baseline.emotions.get(key, 0.0)) * weight
                for key, weight in EMOTION_WEIGHTS.items()
            )
            components.append((emotion_delta, emotion_weight))
        if frame.vocal is not None:
            components.append((frame.vocal, vocal_weight))
        if components:
            total_weight = sum(weight for _, weight in components)
            deltas.append(sum(delta * weight for delta, weight in components) / total_weight)

    if not deltas:
        return ReactionScore(
            score=0.5,
            confidence=0.0,
            sentiment=Sentiment.NEUTRAL,
            window_start=frames[0].timestamp,
            window_end=frames[-1].timestamp,
            frame_count=len(frames),
        )

    score = max(0.0, min(1.0, 0.5 + (sum(deltas) / len(deltas))))
    sentiment = Sentiment.NEUTRAL
    if score >= 0.6:
        sentiment = Sentiment.POSITIVE
    elif score <= 0.4:
        sentiment = Sentiment.NEGATIVE
    return ReactionScore(
        score=round(score, 3),
        confidence=round(min(1.0, len(deltas) / len(frames)), 3),
        sentiment=sentiment,
        window_start=frames[0].timestamp,
        window_end=frames[-1].timestamp,
        frame_count=len(frames),
        source=frames[-1].source,
    )


def cli_to_reaction_score(feedback: str) -> ReactionScore:
    now = time.time()
    mapping = {
        "like": (0.85, Sentiment.POSITIVE),
        "dislike": (0.15, Sentiment.NEGATIVE),
        "meh": (0.5, Sentiment.NEUTRAL),
    }
    key = feedback.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown feedback: {feedback!r}. Use like/dislike/meh.")
    score, sentiment = mapping[key]
    return ReactionScore(
        score=score,
        confidence=1.0,
        sentiment=sentiment,
        window_start=now,
        window_end=now,
        frame_count=1,
        source=SignalSource.CLI,
    )
