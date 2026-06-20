# FER2013 CNN sentiment integration

## Summary

Replace the geometric expression proxy in `webcam.py` with a pre-trained FER2013 CNN (`fer` pip package) for real facial emotion classification. Map the 7-class emotion probability distribution to the existing 0.0-1.0 sentiment score. The change is scoped to the face signal channel only — all downstream aggregation, windowing, reactor logic, and Redis storage remain unchanged.

## Problem

The current `_expression_score()` in `src/backend/webcam.py` computes a crude engagement proxy from two MediaPipe landmark distances (mouth openness 60%, eyebrow raise 40%). It cannot distinguish a smile from a yawn, surprise from disgust, or any actual emotional state. This makes the reaction pipeline unreliable for the DJ agent's sentiment-driven queue decisions (FR-5, FR-7, FR-10).

## Approach

Use the `fer` Python package, which wraps a Keras CNN trained on FER2013 (35,887 labeled 48x48 grayscale facial expression images across 7 emotion classes). It returns per-frame probability distributions over all 7 emotions, runs ~50ms per frame on CPU, and is stable for long-running sessions.

### Why `fer` over alternatives

- **vs DeepFace**: DeepFace wraps multiple backends (age, gender, recognition) — overkill. Slower per-frame (~200ms). Heavier dependency.
- **vs standalone checkpoint**: More manual work for preprocessing, inference wrapper, and maintenance. `fer` handles this out of the box.
- **vs training from scratch**: FER2013 training takes 30-60min on GPU and wouldn't meaningfully outperform the pre-trained model for our 7-class task at a hackathon.

## Architecture

### Current flow

```
webcam frame → MediaPipe landmarks → _expression_score(mouth+brow) → ReactionFrame.face (0.0-1.0)
```

### Proposed flow

```
webcam frame → MediaPipe FaceLandmarker (kept for presence detection)
            → FER CNN (cropped face region) → 7-class emotion probabilities
            → sentiment_score() → ReactionFrame.face (0.0-1.0)
                                → ReactionFrame.emotions (raw probabilities)
```

MediaPipe stays for face detection and presence. FER runs on the detected face crop to classify emotion. The `face` field on `ReactionFrame` keeps its 0.0-1.0 range but now represents actual sentiment.

## Emotion-to-sentiment mapping

FER outputs a probability distribution per frame:

```python
{"happy": 0.65, "surprise": 0.12, "neutral": 0.15, "sad": 0.04, "angry": 0.02, "disgust": 0.01, "fear": 0.01}
```

### Weighted contribution (not argmax)

Using weighted contribution preserves nuance. A frame at 40% happy / 35% neutral / 25% sad is meaningfully different from 90% happy. Argmax would collapse both to "happy".

```python
EMOTION_WEIGHTS = {
    "happy":    +1.0,
    "surprise": +0.5,   # mild positive — can go either way per P2
    "neutral":   0.0,
    "sad":      -0.7,
    "angry":    -1.0,
    "disgust":  -1.0,
    "fear":     -0.5,
}

def sentiment_score(emotions: dict[str, float]) -> float:
    """Map FER emotion probabilities to 0.0-1.0 sentiment score."""
    raw = sum(prob * EMOTION_WEIGHTS[emo] for emo, prob in emotions.items())
    return max(0.0, min(1.0, 0.5 + raw * 0.5))
```

Maps the weighted sum from [-1, 1] to [0, 1].

### Sentiment buckets (unchanged from existing thresholds)

- score >= 0.6 → POSITIVE
- score <= 0.4 → NEGATIVE
- else → NEUTRAL

These thresholds are already defined in `aggregate_window()` in `reaction.py` and remain unchanged.

### Confidence from distribution entropy

A peaked distribution (one emotion dominates) = high confidence. A flat distribution = low confidence. This aligns with P3: anchor on strong signals, distrust weak ones.

```python
def emotion_confidence(emotions: dict[str, float]) -> float:
    """Confidence from how peaked the emotion distribution is."""
    probs = [p for p in emotions.values() if p > 0]
    if not probs:
        return 0.0
    max_prob = max(probs)
    # Scale: 1/7 (uniform) → 0.0 confidence, 1.0 (certain) → 1.0 confidence
    return max(0.0, min(1.0, (max_prob - 1/7) / (1 - 1/7)))
```

## File changes

### `src/backend/reaction.py`

Add one optional field to `ReactionFrame`:

```python
@dataclass
class ReactionFrame:
    timestamp: float = field(default_factory=time.time)
    presence: float | None = None
    movement: float | None = None
    face: float | None = None
    emotions: dict[str, float] | None = None  # NEW: raw FER probabilities
    playback: float | None = None
    vocal: float | None = None
    source: SignalSource = SignalSource.WEBCAM
```

No other changes to `reaction.py`. `aggregate_window()`, `capture_baseline()`, `Baseline`, `ReactionScore` are all untouched.

### `src/backend/webcam.py`

1. **Remove** `_expression_score()` function and its landmark constants (`_UPPER_LIP`, `_LOWER_LIP`, `_LEFT_BROW`, `_LEFT_EYE_CORNER`).

2. **Add** `sentiment_score()` and `emotion_confidence()` functions (defined above).

3. **Add** FER detector initialization in `WebcamWorker.__init__()`:

```python
from fer import FER
# ...
self._fer_detector = FER(mtcnn=False)  # use OpenCV cascade, faster for 1fps
```

Initialized once, reused across all frames. No per-frame model loading.

4. **Update** `_capture_loop()` — after MediaPipe confirms face presence:
   - Crop the face region from the BGR frame using MediaPipe's bounding box
   - Pass the crop to `self._fer_detector.detect_emotions()`
   - Call `sentiment_score()` on the top result's emotion dict
   - Store both score and raw probabilities on `ReactionFrame`

```python
# In _capture_loop, replacing the current face_score computation:
face_score: float | None = None
emotions: dict[str, float] | None = None

if face_detected:
    # Get face bounding box from MediaPipe landmarks
    xs = [lm.x for lm in results.face_landmarks[0]]
    ys = [lm.y for lm in results.face_landmarks[0]]
    x1 = max(0, int(min(xs) * w) - 10)
    y1 = max(0, int(min(ys) * h) - 10)
    x2 = min(w, int(max(xs) * w) + 10)
    y2 = min(h, int(max(ys) * h) + 10)
    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size > 0:
        fer_results = self._fer_detector.detect_emotions(face_crop)
        if fer_results:
            emotions = fer_results[0]["emotions"]
            face_score = sentiment_score(emotions)
```

5. **Update** `ReactionFrame` construction to include `emotions`:

```python
reaction_frame = ReactionFrame(
    timestamp=time.time(),
    presence=presence,
    movement=movement,
    face=face_score,
    emotions=emotions,  # NEW
    source=SignalSource.WEBCAM,
)
```

### `src/backend/test_face.py`

Update the display to show detected emotion alongside the existing overlay:

```python
if f.emotions:
    top_emotion = max(f.emotions, key=f.emotions.get)
    emo_text = f"emotion={top_emotion} ({f.emotions[top_emotion]:.2f})"
    cv2.putText(frame, emo_text, (10, 90), ...)
```

### `src/backend/pyproject.toml`

Add dependency:

```toml
"fer>=22.5.0",
```

### `src/backend/store.py`

No changes. The `emotions` dict flows through `ReactionFrame` serialization as-is via `json.dumps()` in `store_reaction_frame()`.

### `src/backend/reactor.py`

No changes. Reactor reads `ReactionFrame.face` which keeps its 0.0-1.0 contract.

## What stays unchanged

- `reaction.py`: `aggregate_window()`, `capture_baseline()`, `Baseline`, `ReactionScore`, `TrackReaction`, `cli_to_reaction_score()`
- `reactor.py`: entire `Reactor` class
- `store.py`: all Redis storage functions
- `main.py`: FastAPI app
- All downstream sentiment thresholds (0.4/0.6 boundaries)
- The baseline-delta pattern (P3) — baseline captures neutral `face` score at session start, subsequent frames are scored as deltas

## Long-session stability

- FER detector initialized once in `__init__`, never reloaded
- No GPU memory accumulation (CPU inference)
- ~50ms per frame at 1fps = 5% CPU overhead sustained
- Existing `deque(maxlen=120)` caps frame buffer memory
- No new persistent state or growing data structures

## Dependencies

- `fer>=22.5.0` (adds ~15MB; pulls in TensorFlow/Keras if not present)
- TensorFlow is the heaviest transitive dependency (~500MB). Since the project already uses MediaPipe (which depends on TF), this should not add significant new weight.
- If TF becomes a problem at install time, `fer` also supports an `onnxruntime` backend (~50MB) as a fallback.

## PRD alignment

| Requirement | How this design addresses it |
|---|---|
| FR-5 (fuse presence, movement, face, playback, vocal) | Face channel now carries real emotion signal; fusion logic unchanged |
| FR-7 (context-dependent interpretation) | Agent still interprets scores in context via `get_session_context`; we provide better raw signal |
| P2 (never read signal in isolation) | Emotion probabilities are one input to multi-modal fusion, not a standalone decision |
| P3 (calibrate per person, baseline) | Baseline captures neutral face sentiment at session start; deltas computed in `aggregate_window` |
| P4 (fuse multiple modalities) | Face is still one channel alongside presence, movement, playback, vocal |
| P6 (never block playback) | 50ms inference at 1fps in background thread; no blocking |
| P7 (privacy: derive and discard) | Only derived scores and emotion probabilities stored, never raw frames |
