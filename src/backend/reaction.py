"""Reaction data model for ClaudeDJ mood detection.

Follows the Pulse-informed fuse → window → interpret pattern from the PRD.
Raw per-second frames are aggregated over a mid-song window into a single
engagement score. The agent interprets the score relative to track context.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

# Weights for engagement scoring from the two-state emotion output.
# Shared by reaction aggregation and webcam emotion processing.
EMOTION_WEIGHTS: dict[str, float] = {
    "happy": 1.0,
    "neutral": 0.0,
    "disinterested": -0.5,
}

# Mapping from raw 7-class model output to collapsed three-state emotions.
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
    """Confidence from how peaked the collapsed emotion distribution is.
    Uses max probability relative to uniform baseline for the number of categories present.
    """
    probs = [p for p in emotions.values() if p > 0]
    if not probs:
        return 0.0
    n = len(emotions) if len(emotions) > 1 else 3
    uniform = 1.0 / n
    max_prob = max(probs)
    return round(max(0.0, min(1.0, (max_prob - uniform) / (1.0 - uniform))), 3)


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
    """Head orientation in degrees, estimated from face landmarks."""

    yaw: float = 0.0    # left/right turn (-90 to +90)
    pitch: float = 0.0  # up/down nod (-90 to +90)
    roll: float = 0.0   # head tilt (-90 to +90)


@dataclass
class LandmarkExpression:
    """Expression features computed directly from face landmark geometry.

    More reliable than CNN emotion classification for subtle expressions
    because landmarks are pose-invariant and lighting-invariant.
    """

    smile: float = 0.0       # 0.0 (neutral) to 1.0 (big smile) — lip corner rise
    mouth_open: float = 0.0  # 0.0 (closed) to 1.0 (wide open) — singing along
    ear: float = 0.0         # Eye Aspect Ratio — 0.0 (closed) to ~0.4 (wide open)
    brow_height: float = 0.5  # 0.0 (lowered/furrowed) to 1.0 (raised) — interest vs focus


@dataclass
class TrackContext:
    """Track metadata that modulates how reactions are interpreted (FR-7, P2).
    Energy and valence determine how emotions are collapsed:
    - Energy: how movement is weighted (high-energy → movement is positive)
    - Valence: which emotions indicate engagement vs disengagement
      (sad face during a sad song = engaged, not disinterested)
    """
    energy: float = 0.5   # 0.0 (ballad) to 1.0 (high-energy banger)
    valence: float = 0.5  # 0.0 (sad/dark) to 1.0 (happy/bright)
    cluster: str | None = None


def _context_target(emotion: str, energy: float, valence: float) -> str:
    """Determine collapse target for a raw emotion given track context.

    Congruent emotions (matching the track's mood) → "happy" (engaged).
    Incongruent emotions → "disinterested".
    """
    # Always engaged regardless of context
    if emotion in ("happy", "surprise"):
        return "happy"
    if emotion == "neutral":
        return "neutral"
    # Always disinterested regardless of context
    if emotion == "disgust":
        return "disinterested"
    # Context-dependent
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
    """Collapse raw 7-class emotions to 3-state using track context.

    When track context is available, emotions that are congruent with
    the track's mood are treated as engagement signals:
    - Sad face during a sad song (low valence) → engaged
    - Angry face during intense music (low valence, high energy) → engaged
    - Fear/intensity during high-energy music → engaged
    - Disgust is always disinterested

    Without context, falls back to the static RAW_TO_COLLAPSED mapping.
    """
    collapsed: dict[str, float] = {k: 0.0 for k in COLLAPSED_KEYS}

    if track_context is None:
        # Static fallback
        for raw_key, score in raw_emotions.items():
            target = RAW_TO_COLLAPSED.get(raw_key, "disinterested")
            collapsed[target] += score
    else:
        for raw_key, score in raw_emotions.items():
            target = _context_target(raw_key, track_context.energy, track_context.valence)
            collapsed[target] += score

    total = sum(collapsed.values())
    if total > 0:
        collapsed = {k: round(v / total, 4) for k, v in collapsed.items()}
    return collapsed


@dataclass
class ReactionFrame:
    """One ~1s sample of raw reaction signals (FR-5).

    Each component is 0.0–1.0. A value of None means that channel
    was unavailable (e.g., no webcam).
    """

    timestamp: float = field(default_factory=time.time)
    presence: float | None = None  # is the listener there?
    movement: float | None = None  # head movement magnitude (pose delta, not frame diff)
    head_pose: HeadPose | None = None  # current head yaw/pitch/roll
    face: float | None = None  # expression engagement score (0-1)
    raw_emotions: dict[str, float] | None = None  # full 7-class ensemble scores
    emotions: dict[str, float] | None = None  # collapsed 3-state scores (0-1)
    dominant_emotion: str | None = None  # top raw emotion label
    emotion_confidence: float | None = None  # how peaked the emotion distribution is
    landmark_expression: LandmarkExpression | None = None  # geometric expression features
    face_area: float | None = None  # face bounding box area (normalized by frame) — lean-in tracking
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
    emotions: dict[str, float] = field(default_factory=lambda: {
        "happy": 0.0, "neutral": 1.0, "disinterested": 0.0,
    })
    landmark_smile: float = 0.0
    landmark_mouth: float = 0.0
    landmark_ear: float = 0.3   # typical neutral EAR
    landmark_brow: float = 0.5  # neutral brow position
    face_area: float = 0.0     # baseline face bounding box area
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
    emotion_sums: dict[str, float] = {}
    emotion_count = 0
    lm_smile_sum = 0.0
    lm_mouth_sum = 0.0
    lm_ear_sum = 0.0
    lm_brow_sum = 0.0
    lm_count = 0
    area_sum = 0.0
    area_count = 0
    count = 0

    for f in frames:
        if f.movement is not None:
            avg_movement += f.movement
        if f.face is not None:
            avg_face += f.face
        if f.emotions is not None:
            for k, v in f.emotions.items():
                emotion_sums[k] = emotion_sums.get(k, 0.0) + v
            emotion_count += 1
        if f.landmark_expression is not None:
            lm_smile_sum += f.landmark_expression.smile
            lm_mouth_sum += f.landmark_expression.mouth_open
            lm_ear_sum += f.landmark_expression.ear
            lm_brow_sum += f.landmark_expression.brow_height
            lm_count += 1
        if f.face_area is not None:
            area_sum += f.face_area
            area_count += 1
        count += 1

    avg_emotions = {
        k: v / max(emotion_count, 1) for k, v in emotion_sums.items()
    } if emotion_sums else Baseline().emotions

    return Baseline(
        presence=1.0,
        movement=avg_movement / max(count, 1),
        face=avg_face / max(count, 1),
        emotions=avg_emotions,
        landmark_smile=lm_smile_sum / max(lm_count, 1),
        landmark_mouth=lm_mouth_sum / max(lm_count, 1),
        landmark_ear=lm_ear_sum / max(lm_count, 1) if lm_count > 0 else 0.3,
        landmark_brow=lm_brow_sum / max(lm_count, 1) if lm_count > 0 else 0.5,
        face_area=area_sum / max(area_count, 1),
        captured_at=time.time(),
        frame_count=count,
    )


def _analyze_head_patterns(
    frames: list[ReactionFrame],
    track_context: TrackContext | None,
) -> float:
    """Analyze head movement patterns across a window of frames.

    Returns a bonus/penalty delta for engagement scoring:
    - Pitch oscillation with moderate magnitude → rhythmic nodding → positive
    - Sustained high absolute yaw → looking away → negative
    - Stillness during high-energy track → slight negative
    """
    poses = [f.head_pose for f in frames if f.head_pose is not None]
    if len(poses) < 3:
        return 0.0

    # Pitch deltas — sign changes indicate nodding oscillation
    pitch_deltas = [poses[i + 1].pitch - poses[i].pitch for i in range(len(poses) - 1)]
    sign_changes = sum(
        1 for i in range(len(pitch_deltas) - 1)
        if pitch_deltas[i] * pitch_deltas[i + 1] < 0
    )
    oscillation_rate = sign_changes / max(len(pitch_deltas) - 1, 1)
    pitch_magnitude = sum(abs(d) for d in pitch_deltas) / len(pitch_deltas)
    # Nodding: high oscillation rate + moderate magnitude (5° avg = strong nod)
    nod_score = oscillation_rate * min(pitch_magnitude / 5.0, 1.0)

    # Looking away: sustained high absolute yaw
    mean_abs_yaw = sum(abs(p.yaw) for p in poses) / len(poses)
    looking_away = max(0.0, (mean_abs_yaw - 20.0) / 40.0)  # >20° starts penalizing

    bonus = nod_score * 0.15 - looking_away * 0.2

    # Stillness during high-energy track = possible disengagement
    if track_context is not None and track_context.energy > 0.7:
        avg_movement = sum(f.movement or 0.0 for f in frames) / len(frames)
        if avg_movement < 0.05:
            bonus -= 0.05

    return max(-0.2, min(0.15, bonus))


def aggregate_window(
    frames: list[ReactionFrame],
    baseline: Baseline,
    track_context: TrackContext | None = None,
) -> ReactionScore:
    """Aggregate frames over a window into one ReactionScore (FR-6).

    Scores are computed as deltas from baseline (P3).
    Components are fused with context-aware weighting (FR-7, P2).

    Key improvements over naive aggregation:
    - Emotion delta is gated on confidence (>0.3) to avoid noise
    - Landmark expression features (smile, mouth opening) are direct engagement signals
    - Face area changes detect lean-in/lean-back
    - Head movement patterns (nodding, looking away) analyzed across the window
    """
    if not frames:
        return ReactionScore(
            score=0.5,
            confidence=0.0,
            sentiment=Sentiment.NEUTRAL,
        )

    # Context-aware component weights
    movement_weight = 1.0
    face_weight = 1.0
    emotion_weight = 1.0
    vocal_weight = 1.0
    landmark_weight = 1.2   # landmarks are more reliable than CNN for subtle expressions
    proximity_weight = 0.5  # lean-in is a supporting signal

    if track_context is not None:
        energy = track_context.energy
        movement_weight = 0.5 + energy       # high-energy → movement matters more
        face_weight = 1.5 - 0.5 * energy     # low-energy → facial expression matters more

    # Analyze head movement patterns across the full window
    head_bonus = _analyze_head_patterns(frames, track_context)

    deltas: list[float] = []
    for f in frames:
        components: list[tuple[float, float]] = []

        if f.movement is not None:
            components.append((f.movement - baseline.movement, movement_weight))

        if f.face is not None:
            components.append((f.face - baseline.face, face_weight))

        # Confidence-gated emotion scoring: only include emotion delta
        # when the distribution is peaked enough to be meaningful.
        emo_dist = None
        if f.raw_emotions is not None and track_context is not None:
            emo_dist = context_aware_collapse(f.raw_emotions, track_context)
        elif f.emotions is not None:
            emo_dist = f.emotions

        conf = f.emotion_confidence if f.emotion_confidence is not None else 0.0
        if emo_dist is not None and conf > 0.3:
            emotion_delta = sum(
                (emo_dist.get(k, 0.0) - baseline.emotions.get(k, 0.0)) * w
                for k, w in EMOTION_WEIGHTS.items()
            )
            # Scale contribution by confidence — peaked distributions count more
            components.append((emotion_delta * conf, emotion_weight))

        # Landmark expression signals — more reliable than CNN for micro-expressions
        if f.landmark_expression is not None:
            lm = f.landmark_expression
            smile_delta = lm.smile - baseline.landmark_smile
            mouth_delta = lm.mouth_open - baseline.landmark_mouth
            # Micro-smile: even small positive deltas are engagement
            if smile_delta > 0.05:
                components.append((smile_delta, landmark_weight))
            # Mouth opening above baseline = singing along / mouthing words
            if mouth_delta > 0.1:
                components.append((mouth_delta * 0.8, landmark_weight))
            # Brow lowering (focus) during low-energy tracks = engagement
            if track_context is not None and track_context.energy < 0.4:
                brow_delta = lm.brow_height - baseline.landmark_brow
                if brow_delta < -0.1:  # brows lowered = focused
                    components.append((abs(brow_delta) * 0.5, landmark_weight * 0.5))

        # Face proximity: increasing area = lean-in = engagement
        if f.face_area is not None and baseline.face_area > 0:
            area_ratio = (f.face_area - baseline.face_area) / baseline.face_area
            if abs(area_ratio) > 0.05:  # >5% change is meaningful
                components.append((area_ratio * 0.5, proximity_weight))

        if f.vocal is not None:
            components.append((f.vocal, vocal_weight))

        if components:
            total_weight = sum(w for _, w in components)
            weighted_sum = sum(d * w for d, w in components)
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
