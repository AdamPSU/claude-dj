from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


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

COLLAPSED_KEYS: list[str] = ["happy", "neutral", "disinterested"]


def emotion_confidence(emotions: dict[str, float]) -> float:
    probs = [probability for probability in emotions.values() if probability > 0]
    if not probs:
        return 0.0
    n = len(emotions) if len(emotions) > 1 else 3
    uniform = 1.0 / n
    max_probability = max(probs)
    return round(max(0.0, min(1.0, (max_probability - uniform) / (1.0 - uniform))), 3)


class Sentiment(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class SignalSource(Enum):
    CLI = "cli"
    WEBCAM = "webcam"
    PLAYBACK = "playback"


@dataclass
class HeadPose:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class LandmarkExpression:
    smile: float = 0.0
    mouth_open: float = 0.0
    ear: float = 0.0
    brow_height: float = 0.5


@dataclass
class TrackContext:
    energy: float = 0.5
    valence: float = 0.5
    cluster: str | None = None


def _context_target(emotion: str, energy: float, valence: float) -> str:
    if emotion in ("happy", "surprise"):
        return "happy"
    if emotion == "neutral":
        return "neutral"
    if emotion == "disgust":
        return "disinterested"
    if emotion == "sad":
        return "happy" if valence < 0.4 else "disinterested"
    if emotion == "angry":
        return "happy" if valence < 0.4 and energy > 0.6 else "disinterested"
    if emotion == "fear":
        return "happy" if energy > 0.6 else "disinterested"
    return "disinterested"


def context_aware_collapse(
    raw_emotions: dict[str, float],
    track_context: TrackContext | None = None,
) -> dict[str, float]:
    collapsed: dict[str, float] = {key: 0.0 for key in COLLAPSED_KEYS}
    if track_context is None:
        for raw_key, score in raw_emotions.items():
            target = RAW_TO_COLLAPSED.get(raw_key, "disinterested")
            collapsed[target] += score
    else:
        for raw_key, score in raw_emotions.items():
            target = _context_target(raw_key, track_context.energy, track_context.valence)
            collapsed[target] += score
    total = sum(collapsed.values())
    if total > 0:
        collapsed = {key: round(value / total, 4) for key, value in collapsed.items()}
    return collapsed


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
    landmark_expression: LandmarkExpression | None = None
    face_area: float | None = None
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
    landmark_smile: float = 0.0
    landmark_mouth: float = 0.0
    landmark_ear: float = 0.3
    landmark_brow: float = 0.5
    face_area: float = 0.0
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


@dataclass
class TrackReaction:
    track_id: str
    frames: list[ReactionFrame] = field(default_factory=list)
    scores: list[ReactionScore] = field(default_factory=list)
    final_sentiment: Sentiment | None = None
    final_score: float | None = None


def cli_to_reaction_score(feedback: str) -> ReactionScore:
    now = time.time()
    mapping: dict[str, tuple[float, Sentiment]] = {
        "like": (0.85, Sentiment.POSITIVE),
        "dislike": (0.15, Sentiment.NEGATIVE),
        "meh": (0.5, Sentiment.NEUTRAL),
    }
    feedback_lower = feedback.strip().lower()
    if feedback_lower not in mapping:
        raise ValueError(f"Unknown feedback: {feedback!r}. Use like/dislike/meh.")
    score_value, sentiment = mapping[feedback_lower]
    return ReactionScore(
        score=score_value,
        confidence=1.0,
        sentiment=sentiment,
        window_start=now,
        window_end=now,
        frame_count=1,
        source=SignalSource.CLI,
    )


def capture_baseline(frames: list[ReactionFrame]) -> Baseline:
    if not frames:
        return Baseline()

    avg_movement = 0.0
    avg_face = 0.0
    emotion_sums: dict[str, float] = {}
    emotion_count = 0
    landmark_smile_sum = 0.0
    landmark_mouth_sum = 0.0
    landmark_ear_sum = 0.0
    landmark_brow_sum = 0.0
    landmark_count = 0
    area_sum = 0.0
    area_count = 0
    count = 0

    for frame in frames:
        if frame.movement is not None:
            avg_movement += frame.movement
        if frame.face is not None:
            avg_face += frame.face
        if frame.emotions is not None:
            for key, value in frame.emotions.items():
                emotion_sums[key] = emotion_sums.get(key, 0.0) + value
            emotion_count += 1
        if frame.landmark_expression is not None:
            landmark_smile_sum += frame.landmark_expression.smile
            landmark_mouth_sum += frame.landmark_expression.mouth_open
            landmark_ear_sum += frame.landmark_expression.ear
            landmark_brow_sum += frame.landmark_expression.brow_height
            landmark_count += 1
        if frame.face_area is not None:
            area_sum += frame.face_area
            area_count += 1
        count += 1

    avg_emotions = (
        {key: value / max(emotion_count, 1) for key, value in emotion_sums.items()}
        if emotion_sums
        else Baseline().emotions
    )

    return Baseline(
        presence=1.0,
        movement=avg_movement / max(count, 1),
        face=avg_face / max(count, 1),
        emotions=avg_emotions,
        landmark_smile=landmark_smile_sum / max(landmark_count, 1),
        landmark_mouth=landmark_mouth_sum / max(landmark_count, 1),
        landmark_ear=landmark_ear_sum / max(landmark_count, 1) if landmark_count > 0 else 0.3,
        landmark_brow=landmark_brow_sum / max(landmark_count, 1) if landmark_count > 0 else 0.5,
        face_area=area_sum / max(area_count, 1),
        captured_at=time.time(),
        frame_count=count,
    )


def _analyze_head_patterns(frames: list[ReactionFrame], track_context: TrackContext | None) -> float:
    poses = [frame.head_pose for frame in frames if frame.head_pose is not None]
    if len(poses) < 3:
        return 0.0

    pitch_deltas = [poses[index + 1].pitch - poses[index].pitch for index in range(len(poses) - 1)]
    sign_changes = sum(
        1
        for index in range(len(pitch_deltas) - 1)
        if pitch_deltas[index] * pitch_deltas[index + 1] < 0
    )
    oscillation_rate = sign_changes / max(len(pitch_deltas) - 1, 1)
    pitch_magnitude = sum(abs(delta) for delta in pitch_deltas) / len(pitch_deltas)
    nod_score = oscillation_rate * min(pitch_magnitude / 5.0, 1.0)
    mean_abs_yaw = sum(abs(pose.yaw) for pose in poses) / len(poses)
    looking_away = max(0.0, (mean_abs_yaw - 20.0) / 40.0)

    bonus = nod_score * 0.15 - looking_away * 0.2
    if track_context is not None and track_context.energy > 0.7:
        avg_movement = sum(frame.movement or 0.0 for frame in frames) / len(frames)
        if avg_movement < 0.05:
            bonus -= 0.05
    return max(-0.2, min(0.15, bonus))


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
    landmark_weight = 1.2
    proximity_weight = 0.5

    if track_context is not None:
        movement_weight = 0.5 + track_context.energy
        face_weight = 1.5 - (0.5 * track_context.energy)

    head_bonus = _analyze_head_patterns(frames, track_context)
    deltas: list[float] = []
    for frame in frames:
        components: list[tuple[float, float]] = []

        if frame.movement is not None:
            components.append((frame.movement - baseline.movement, movement_weight))
        if frame.face is not None:
            components.append((frame.face - baseline.face, face_weight))

        emotion_distribution = None
        if frame.raw_emotions is not None and track_context is not None:
            emotion_distribution = context_aware_collapse(frame.raw_emotions, track_context)
        elif frame.emotions is not None:
            emotion_distribution = frame.emotions

        confidence = frame.emotion_confidence if frame.emotion_confidence is not None else 0.0
        if emotion_distribution is not None and confidence > 0.3:
            emotion_delta = sum(
                (emotion_distribution.get(key, 0.0) - baseline.emotions.get(key, 0.0)) * weight
                for key, weight in EMOTION_WEIGHTS.items()
            )
            components.append((emotion_delta * confidence, emotion_weight))

        if frame.landmark_expression is not None:
            landmark = frame.landmark_expression
            smile_delta = landmark.smile - baseline.landmark_smile
            mouth_delta = landmark.mouth_open - baseline.landmark_mouth
            if smile_delta > 0.05:
                components.append((smile_delta, landmark_weight))
            if mouth_delta > 0.1:
                components.append((mouth_delta * 0.8, landmark_weight))
            if track_context is not None and track_context.energy < 0.4:
                brow_delta = landmark.brow_height - baseline.landmark_brow
                if brow_delta < -0.1:
                    components.append((abs(brow_delta) * 0.5, landmark_weight * 0.5))

        if frame.face_area is not None and baseline.face_area > 0:
            area_ratio = (frame.face_area - baseline.face_area) / baseline.face_area
            if abs(area_ratio) > 0.05:
                components.append((area_ratio * 0.5, proximity_weight))

        if frame.vocal is not None:
            components.append((frame.vocal, vocal_weight))

        if components:
            total_weight = sum(weight for _, weight in components)
            weighted_sum = sum(delta * weight for delta, weight in components)
            deltas.append(weighted_sum / total_weight)

    if not deltas:
        return ReactionScore(
            score=0.5,
            confidence=0.0,
            sentiment=Sentiment.NEUTRAL,
            window_start=frames[0].timestamp,
            window_end=frames[-1].timestamp,
            frame_count=len(frames),
        )

    raw_delta = sum(deltas) / len(deltas) + head_bonus
    score = max(0.0, min(1.0, 0.5 + raw_delta))
    confidence = min(1.0, len(deltas) / max(len(frames), 1))
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
