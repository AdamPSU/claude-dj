# Backend Emotion Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the real-time video ML emotion detection backend so the DJ agent can reliably determine whether a listener is engaged, disinterested, or neutral — with proper tests, context-conditioned interpretation, confidence weighting, and API endpoints.

**Architecture:** The webcam worker captures frames at ~1fps in a background thread, runs a ViT-FER + DeepFace ensemble to produce 7-class emotions collapsed to 3-state (happy/neutral/disinterested), smoothed temporally. The reactor pipeline fuses webcam + CLI signals into windowed engagement scores. FastAPI endpoints expose the reaction data to the MCP server / agent. All scoring is context-conditioned: the same reaction means different things depending on the track's energy level.

**Tech Stack:** Python 3.11+, pytest, FastAPI, OpenCV, MediaPipe, HuggingFace transformers (ViT-FER), DeepFace, Redis, numpy

**Existing files (read these first):**
- `src/backend/reaction.py` — Data models: ReactionFrame, Baseline, ReactionScore, aggregate_window(), capture_baseline()
- `src/backend/webcam.py` — WebcamWorker, ViT-FER + DeepFace ensemble, head pose, CLAHE, temporal smoothing
- `src/backend/reactor.py` — Reactor: fuses webcam + CLI, produces windowed scores and trends
- `src/backend/store.py` — Redis storage: reaction frames, session state, memory, session context bundle
- `src/backend/main.py` — FastAPI app (currently only health check)
- `src/backend/test_face.py` — Manual webcam test (has a bug: wrong _ensemble_emotions call)
- `src/backend/test_fer.py` — Manual model validation script

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `src/backend/tests/test_emotion_pipeline.py` | Unit tests for sentiment scoring, ensemble blending, smoothing, confidence | Create |
| `src/backend/tests/test_reaction.py` | Unit tests for aggregate_window, capture_baseline, cli_to_reaction_score | Create |
| `src/backend/tests/test_reactor.py` | Unit tests for Reactor (webcam disabled, CLI-only mode) | Create |
| `src/backend/tests/test_api.py` | Unit tests for FastAPI reaction endpoints | Create |
| `src/backend/tests/__init__.py` | Package init | Create |
| `src/backend/tests/conftest.py` | Shared fixtures | Create |
| `src/backend/reaction.py` | Add emotion_confidence(), add track_context param to aggregate_window | Modify |
| `src/backend/webcam.py` | Add emotion_confidence to ReactionFrame construction | Modify |
| `src/backend/reactor.py` | Accept track context for context-conditioned scoring | Modify |
| `src/backend/main.py` | Add reaction API endpoints, wire reactor lifecycle | Modify |
| `src/backend/test_face.py` | Fix broken _ensemble_emotions call | Modify |

---

### Task 1: Fix test_face.py broken API call

**Files:**
- Modify: `src/backend/test_face.py:99-104`

The `_ensemble_emotions()` function returns a `tuple[dict | None, dict | None]` (raw_7class, collapsed_3state), but `test_face.py` unpacks it as a single value. This causes the smoothing and display to fail silently.

- [ ] **Step 1: Fix the _ensemble_emotions call and downstream usage**

In `src/backend/test_face.py`, replace lines 99-104:

```python
                # OLD (broken):
                # raw_ensemble = _ensemble_emotions(vit_emos, df_emos)
                # if raw_ensemble:
                #     emotions = _smooth_emotions(raw_ensemble, prev_smoothed)
                #     prev_smoothed = emotions
                #     dominant = max(emotions, key=emotions.get)
                #     face_score = _engagement_score(emotions)

                # NEW (correct):
                raw_ensemble, collapsed_ensemble = _ensemble_emotions(vit_emos, df_emos)
                if collapsed_ensemble:
                    emotions = _smooth_emotions(collapsed_ensemble, prev_smoothed)
                    prev_smoothed = emotions
                    face_score = _engagement_score(emotions)
                if raw_ensemble:
                    dominant = max(raw_ensemble, key=raw_ensemble.get)
```

- [ ] **Step 2: Verify the fix runs**

Run: `cd src/backend && python test_face.py`
Expected: Window opens showing webcam with emotion overlay. Press 'q' to quit. No Python errors in console.

- [ ] **Step 3: Commit**

```bash
git add src/backend/test_face.py
git commit -m "fix: correct _ensemble_emotions tuple unpacking in test_face.py"
```

---

### Task 2: Add emotion confidence (distribution entropy) to reaction pipeline

**Files:**
- Modify: `src/backend/reaction.py` (add `emotion_confidence` field to ReactionFrame, add `emotion_confidence()` function)
- Modify: `src/backend/webcam.py` (compute and store confidence per frame)
- Create: `src/backend/tests/__init__.py`
- Create: `src/backend/tests/conftest.py`
- Create: `src/backend/tests/test_emotion_pipeline.py`

Emotion confidence measures how peaked the distribution is. A frame where one emotion dominates (e.g., 80% happy) is high confidence. A flat distribution (all ~14%) is low confidence. This lets aggregate_window weight frames appropriately (P3: anchor on strong signals).

- [ ] **Step 1: Create test infrastructure**

Create `src/backend/tests/__init__.py` (empty file).

Create `src/backend/tests/conftest.py`:

```python
"""Shared test fixtures for ClaudeDJ backend tests."""

import sys
from pathlib import Path

# Add src/backend to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 2: Write failing tests for emotion_confidence**

Create `src/backend/tests/test_emotion_pipeline.py`:

```python
"""Tests for emotion pipeline: confidence, scoring, ensemble, smoothing."""

import pytest
from reaction import ReactionFrame, emotion_confidence


class TestEmotionConfidence:
    """emotion_confidence: distribution entropy → 0.0-1.0 confidence."""

    def test_peaked_distribution_high_confidence(self):
        """One emotion at 90% → confidence near 1.0."""
        emotions = {"happy": 0.9, "neutral": 0.05, "disinterested": 0.05}
        conf = emotion_confidence(emotions)
        assert conf >= 0.8
        assert conf <= 1.0

    def test_uniform_distribution_low_confidence(self):
        """All emotions equal → confidence near 0.0."""
        emotions = {"happy": 0.333, "neutral": 0.333, "disinterested": 0.334}
        conf = emotion_confidence(emotions)
        assert conf <= 0.15

    def test_empty_distribution_zero_confidence(self):
        """No probabilities → 0.0 confidence."""
        conf = emotion_confidence({})
        assert conf == 0.0

    def test_all_zeros_zero_confidence(self):
        """All zeros → 0.0 confidence."""
        emotions = {"happy": 0.0, "neutral": 0.0, "disinterested": 0.0}
        conf = emotion_confidence(emotions)
        assert conf == 0.0

    def test_moderate_distribution_moderate_confidence(self):
        """60/30/10 split → moderate confidence."""
        emotions = {"happy": 0.6, "neutral": 0.3, "disinterested": 0.1}
        conf = emotion_confidence(emotions)
        assert 0.3 <= conf <= 0.7

    def test_returns_float_in_range(self):
        """Always returns a float in [0.0, 1.0]."""
        emotions = {"happy": 0.5, "neutral": 0.3, "disinterested": 0.2}
        conf = emotion_confidence(emotions)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd src/backend && python -m pytest tests/test_emotion_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'emotion_confidence' from 'reaction'`

- [ ] **Step 4: Implement emotion_confidence in reaction.py**

Add to `src/backend/reaction.py` after the `COLLAPSED_KEYS` definition:

```python
def emotion_confidence(emotions: dict[str, float]) -> float:
    """Confidence from how peaked the collapsed emotion distribution is.

    A peaked distribution (one emotion dominates) = high confidence.
    A flat distribution = low confidence. Aligns with P3: anchor on
    strong signals, distrust weak ones.

    Uses max probability relative to uniform baseline for the number
    of categories present.
    """
    probs = [p for p in emotions.values() if p > 0]
    if not probs:
        return 0.0
    n = len(emotions) if len(emotions) > 1 else 3  # default to 3-state
    uniform = 1.0 / n
    max_prob = max(probs)
    return round(max(0.0, min(1.0, (max_prob - uniform) / (1.0 - uniform))), 3)
```

- [ ] **Step 5: Add emotion_confidence field to ReactionFrame**

In `src/backend/reaction.py`, add the field to the `ReactionFrame` dataclass after `dominant_emotion`:

```python
    emotion_confidence: float | None = None  # how peaked the emotion distribution is
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd src/backend && python -m pytest tests/test_emotion_pipeline.py -v`
Expected: All 6 tests PASS

- [ ] **Step 7: Wire emotion_confidence into WebcamWorker**

In `src/backend/webcam.py`, add the import at the top:

```python
from reaction import (
    Baseline, COLLAPSED_KEYS, EMOTION_WEIGHTS, HeadPose, RAW_TO_COLLAPSED,
    ReactionFrame, SignalSource, capture_baseline, emotion_confidence,
)
```

In `_capture_loop()`, after `face_score = _engagement_score(collapsed_emotions)` (around line 399), add:

```python
                        face_conf = emotion_confidence(collapsed_emotions)
```

Initialize `face_conf` to `None` alongside the other variables at the top of the face_detected block (around line 345):

```python
                face_conf: float | None = None
```

Update the ReactionFrame construction (around line 406) to include:

```python
                reaction_frame = ReactionFrame(
                    timestamp=time.time(),
                    presence=presence,
                    movement=movement,
                    head_pose=head_pose,
                    face=face_score,
                    raw_emotions=raw_emotions,
                    emotions=collapsed_emotions,
                    dominant_emotion=dominant_emotion,
                    emotion_confidence=face_conf,
                    source=SignalSource.WEBCAM,
                )
```

- [ ] **Step 8: Commit**

```bash
git add src/backend/reaction.py src/backend/webcam.py src/backend/tests/
git commit -m "feat: add emotion confidence scoring from distribution entropy"
```

---

### Task 3: Add context-conditioned interpretation to aggregate_window (FR-7)

**Files:**
- Modify: `src/backend/reaction.py` (add TrackContext dataclass, update aggregate_window signature)
- Modify: `src/backend/reactor.py` (pass track context to aggregate_window)
- Create: `src/backend/tests/test_reaction.py`

The PRD (P2, FR-7) says: "Movement during an up-tempo track signals engagement; stillness during a ballad does not signal dislike." Currently aggregate_window treats all signals identically regardless of track context. This task adds a `TrackContext` that modulates how movement and stillness are interpreted.

- [ ] **Step 1: Write failing tests for context-conditioned aggregation**

Create `src/backend/tests/test_reaction.py`:

```python
"""Tests for reaction aggregation, baseline capture, and context conditioning."""

import time
import pytest
from reaction import (
    Baseline, ReactionFrame, ReactionScore, Sentiment, SignalSource,
    TrackContext, aggregate_window, capture_baseline, cli_to_reaction_score,
    emotion_confidence,
)


def _make_frame(**kwargs) -> ReactionFrame:
    """Helper to create a ReactionFrame with defaults."""
    defaults = {
        "timestamp": time.time(),
        "presence": 1.0,
        "movement": 0.3,
        "face": 0.6,
        "emotions": {"happy": 0.6, "neutral": 0.3, "disinterested": 0.1},
        "source": SignalSource.WEBCAM,
    }
    defaults.update(kwargs)
    return ReactionFrame(**defaults)


class TestAggregateWindowBasic:
    """aggregate_window without track context (existing behavior)."""

    def test_empty_frames_returns_neutral(self):
        score = aggregate_window([], Baseline())
        assert score.sentiment == Sentiment.NEUTRAL
        assert score.confidence == 0.0

    def test_positive_frames_return_positive(self):
        frames = [_make_frame(face=0.8, movement=0.5,
                              emotions={"happy": 0.8, "neutral": 0.1, "disinterested": 0.1})]
        baseline = Baseline(face=0.3, movement=0.1,
                            emotions={"happy": 0.2, "neutral": 0.6, "disinterested": 0.2})
        score = aggregate_window(frames, baseline)
        assert score.score > 0.6
        assert score.sentiment == Sentiment.POSITIVE

    def test_negative_frames_return_negative(self):
        frames = [_make_frame(face=0.1, movement=0.0,
                              emotions={"happy": 0.0, "neutral": 0.2, "disinterested": 0.8})]
        baseline = Baseline(face=0.5, movement=0.3,
                            emotions={"happy": 0.3, "neutral": 0.5, "disinterested": 0.2})
        score = aggregate_window(frames, baseline)
        assert score.score < 0.4
        assert score.sentiment == Sentiment.NEGATIVE


class TestContextConditioned:
    """aggregate_window with TrackContext (FR-7)."""

    def test_high_energy_track_boosts_movement(self):
        """Movement during up-tempo track should boost engagement."""
        frames = [_make_frame(movement=0.6, face=0.5,
                              emotions={"happy": 0.4, "neutral": 0.4, "disinterested": 0.2})]
        baseline = Baseline(movement=0.1, face=0.5,
                            emotions={"happy": 0.4, "neutral": 0.4, "disinterested": 0.2})
        ctx = TrackContext(energy=0.9)

        score_with_ctx = aggregate_window(frames, baseline, track_context=ctx)
        score_without = aggregate_window(frames, baseline)

        # With high-energy context, movement delta should contribute more
        assert score_with_ctx.score > score_without.score

    def test_low_energy_track_does_not_penalize_stillness(self):
        """Stillness during a ballad should not be read as disengagement."""
        frames = [_make_frame(movement=0.05, face=0.5,
                              emotions={"happy": 0.3, "neutral": 0.5, "disinterested": 0.2})]
        baseline = Baseline(movement=0.3, face=0.5,
                            emotions={"happy": 0.3, "neutral": 0.5, "disinterested": 0.2})
        ctx = TrackContext(energy=0.2)

        score_with_ctx = aggregate_window(frames, baseline, track_context=ctx)
        score_without = aggregate_window(frames, baseline)

        # Without context, low movement vs higher baseline = negative delta
        # With low-energy context, movement weight is reduced so score is higher (less negative)
        assert score_with_ctx.score >= score_without.score

    def test_no_context_same_as_before(self):
        """Without TrackContext, behavior is unchanged."""
        frames = [_make_frame()]
        baseline = Baseline()
        score_none = aggregate_window(frames, baseline, track_context=None)
        score_default = aggregate_window(frames, baseline)
        assert score_none.score == score_default.score


class TestCaptureBaseline:
    def test_baseline_from_frames(self):
        frames = [
            _make_frame(movement=0.1, face=0.4,
                        emotions={"happy": 0.2, "neutral": 0.6, "disinterested": 0.2}),
            _make_frame(movement=0.2, face=0.5,
                        emotions={"happy": 0.3, "neutral": 0.5, "disinterested": 0.2}),
        ]
        bl = capture_baseline(frames)
        assert bl.frame_count == 2
        assert 0.1 <= bl.movement <= 0.2
        assert 0.4 <= bl.face <= 0.5

    def test_baseline_empty_frames(self):
        bl = capture_baseline([])
        assert bl.frame_count == 0
        assert bl.movement == 0.0


class TestCliToReactionScore:
    def test_like(self):
        score = cli_to_reaction_score("like")
        assert score.sentiment == Sentiment.POSITIVE
        assert score.confidence == 1.0

    def test_dislike(self):
        score = cli_to_reaction_score("dislike")
        assert score.sentiment == Sentiment.NEGATIVE

    def test_meh(self):
        score = cli_to_reaction_score("meh")
        assert score.sentiment == Sentiment.NEUTRAL

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            cli_to_reaction_score("whatever")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/backend && python -m pytest tests/test_reaction.py -v`
Expected: FAIL — `ImportError: cannot import name 'TrackContext' from 'reaction'`

- [ ] **Step 3: Add TrackContext dataclass to reaction.py**

Add after the `HeadPose` dataclass in `src/backend/reaction.py`:

```python
@dataclass
class TrackContext:
    """Track metadata that modulates how reactions are interpreted (FR-7, P2).

    Energy level determines how movement is weighted: high-energy tracks
    make movement a stronger positive signal, low-energy tracks reduce
    the penalty for stillness.
    """

    energy: float = 0.5  # 0.0 (ballad) to 1.0 (high-energy banger)
    cluster: str | None = None  # current cluster name, for logging
```

- [ ] **Step 4: Update aggregate_window to accept TrackContext**

In `src/backend/reaction.py`, update the `aggregate_window` function signature and body:

```python
def aggregate_window(
    frames: list[ReactionFrame],
    baseline: Baseline,
    track_context: TrackContext | None = None,
) -> ReactionScore:
    """Aggregate frames over a window into one ReactionScore (FR-6).

    Scores are computed as deltas from baseline (P3).
    Components are fused with context-aware weighting (FR-7, P2):
    - High-energy track: movement weight increases (dancing = engaged)
    - Low-energy track: movement weight decreases (stillness != disengaged)
    """
    if not frames:
        return ReactionScore(
            score=0.5,
            confidence=0.0,
            sentiment=Sentiment.NEUTRAL,
        )

    # Context-conditioned component weights (FR-7)
    # Default: equal weight for all components
    movement_weight = 1.0
    face_weight = 1.0
    emotion_weight = 1.0
    vocal_weight = 1.0

    if track_context is not None:
        energy = track_context.energy
        # High energy (0.8-1.0): movement is 1.5x important
        # Low energy (0.0-0.2): movement is 0.5x important
        # Mid energy (0.5): movement is 1.0x (no change)
        movement_weight = 0.5 + energy  # 0.5 at energy=0, 1.5 at energy=1
        # Face/emotion gets slightly more weight when movement is down-weighted
        face_weight = 1.5 - 0.5 * energy  # 1.5 at energy=0, 1.0 at energy=1

    deltas: list[float] = []
    for f in frames:
        components: list[tuple[float, float]] = []  # (delta, weight)
        if f.movement is not None:
            components.append((f.movement - baseline.movement, movement_weight))
        if f.face is not None:
            components.append((f.face - baseline.face, face_weight))
        if f.emotions is not None:
            emotion_delta = sum(
                (f.emotions.get(k, 0.0) - baseline.emotions.get(k, 0.0)) * w
                for k, w in EMOTION_WEIGHTS.items()
            )
            components.append((emotion_delta, emotion_weight))
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd src/backend && python -m pytest tests/test_reaction.py -v`
Expected: All tests PASS

- [ ] **Step 6: Update reactor.py to pass track context**

In `src/backend/reactor.py`, add the import:

```python
from reaction import (
    Baseline,
    ReactionFrame,
    ReactionScore,
    Sentiment,
    SignalSource,
    TrackContext,
    aggregate_window,
    capture_baseline,
    cli_to_reaction_score,
)
```

Add a `_track_context` field to `Reactor.__init__`:

```python
        self._track_context: TrackContext | None = None
```

Add a method to set track context:

```python
    def set_track_context(self, energy: float = 0.5, cluster: str | None = None) -> None:
        """Update the current track context for context-conditioned scoring (FR-7)."""
        self._track_context = TrackContext(energy=energy, cluster=cluster)
```

Update `get_current_score` to pass context to `aggregate_window`:

```python
                return aggregate_window(frames, baseline, track_context=self._track_context)
```

Update `get_trend` similarly — the `aggregate_window` call inside the loop:

```python
                scores.append(aggregate_window(frames, baseline, track_context=self._track_context))
```

- [ ] **Step 7: Commit**

```bash
git add src/backend/reaction.py src/backend/reactor.py src/backend/tests/test_reaction.py
git commit -m "feat: add context-conditioned interpretation for track energy (FR-7)"
```

---

### Task 4: Add unit tests for Reactor (CLI-only mode)

**Files:**
- Create: `src/backend/tests/test_reactor.py`

Tests the Reactor in webcam-disabled mode (CLI feedback only) to validate the fusion, windowing, and trend logic without requiring a camera.

- [ ] **Step 1: Write tests**

Create `src/backend/tests/test_reactor.py`:

```python
"""Tests for Reactor pipeline (CLI-only mode, no webcam)."""

import time
import pytest
from reactor import Reactor
from reaction import Sentiment, TrackContext


class TestReactorCliOnly:
    """Reactor with enable_webcam=False, CLI feedback only."""

    def test_no_data_returns_neutral(self):
        r = Reactor(enable_webcam=False)
        score = r.get_current_score()
        assert score.sentiment == Sentiment.NEUTRAL
        assert score.confidence == 0.0

    def test_like_returns_positive(self):
        r = Reactor(enable_webcam=False)
        r.add_cli_feedback("like")
        score = r.get_current_score()
        assert score.sentiment == Sentiment.POSITIVE
        assert score.confidence == 1.0

    def test_dislike_returns_negative(self):
        r = Reactor(enable_webcam=False)
        r.add_cli_feedback("dislike")
        score = r.get_current_score()
        assert score.sentiment == Sentiment.NEGATIVE

    def test_meh_returns_neutral(self):
        r = Reactor(enable_webcam=False)
        r.add_cli_feedback("meh")
        score = r.get_current_score()
        assert score.sentiment == Sentiment.NEUTRAL
        assert score.confidence == 1.0  # CLI is high confidence

    def test_most_recent_cli_wins(self):
        r = Reactor(enable_webcam=False)
        r.add_cli_feedback("like")
        r.add_cli_feedback("dislike")
        score = r.get_current_score()
        assert score.sentiment == Sentiment.NEGATIVE

    def test_get_summary_shape(self):
        r = Reactor(enable_webcam=False)
        r.add_cli_feedback("like")
        summary = r.get_summary()
        assert "current_score" in summary
        assert "confidence" in summary
        assert "sentiment" in summary
        assert "trend_direction" in summary
        assert summary["sentiment"] == "positive"

    def test_set_track_context(self):
        r = Reactor(enable_webcam=False)
        r.set_track_context(energy=0.9, cluster="reggaeton")
        assert r._track_context is not None
        assert r._track_context.energy == 0.9

    def test_get_trend_returns_list(self):
        r = Reactor(enable_webcam=False)
        trend = r.get_trend(windows=3)
        assert isinstance(trend, list)
        assert len(trend) == 3

    def test_invalid_feedback_raises(self):
        r = Reactor(enable_webcam=False)
        with pytest.raises(ValueError):
            r.add_cli_feedback("love_it")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd src/backend && python -m pytest tests/test_reactor.py -v`
Expected: All 9 tests PASS (these test existing behavior, should pass immediately)

- [ ] **Step 3: Commit**

```bash
git add src/backend/tests/test_reactor.py
git commit -m "test: add Reactor unit tests for CLI-only mode"
```

---

### Task 5: Add FastAPI reaction endpoints and wire reactor lifecycle

**Files:**
- Modify: `src/backend/main.py` (add endpoints, reactor lifecycle)
- Create: `src/backend/tests/test_api.py`

The MCP server and agent need HTTP endpoints to query engagement scores, submit CLI feedback, set track context, and get the full summary. These wrap the Reactor.

- [ ] **Step 1: Write failing tests for API endpoints**

Create `src/backend/tests/test_api.py`:

```python
"""Tests for FastAPI reaction endpoints."""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestReactionEndpoints:
    def test_get_score_default(self, client):
        resp = client.get("/reaction/score")
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "sentiment" in data
        assert "confidence" in data

    def test_post_feedback_like(self, client):
        resp = client.post("/reaction/feedback", json={"feedback": "like"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"] == "positive"

    def test_post_feedback_dislike(self, client):
        resp = client.post("/reaction/feedback", json={"feedback": "dislike"})
        assert resp.status_code == 200
        assert resp.json()["sentiment"] == "negative"

    def test_post_feedback_invalid(self, client):
        resp = client.post("/reaction/feedback", json={"feedback": "love"})
        assert resp.status_code == 400

    def test_get_summary(self, client):
        resp = client.get("/reaction/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_score" in data
        assert "trend_direction" in data

    def test_get_trend(self, client):
        resp = client.get("/reaction/trend")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["scores"], list)

    def test_post_track_context(self, client):
        resp = client.post("/reaction/track-context", json={"energy": 0.8, "cluster": "reggaeton"})
        assert resp.status_code == 200
        assert resp.json()["energy"] == 0.8

    def test_post_track_context_defaults(self, client):
        resp = client.post("/reaction/track-context", json={})
        assert resp.status_code == 200
        assert resp.json()["energy"] == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/backend && python -m pytest tests/test_api.py -v`
Expected: FAIL — 404 for `/reaction/score` and other endpoints

- [ ] **Step 3: Implement reaction endpoints in main.py**

Replace `src/backend/main.py`:

```python
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
    cluster: str | None = None


@app.post("/reaction/track-context")
def post_track_context(req: TrackContextRequest) -> dict:
    """Set current track context for context-conditioned scoring (FR-7)."""
    reactor.set_track_context(energy=req.energy, cluster=req.cluster)
    return {"energy": req.energy, "cluster": req.cluster}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 4: Run API tests to verify they pass**

Run: `cd src/backend && CLAUDEDJ_NO_WEBCAM=1 python -m pytest tests/test_api.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Run all tests together**

Run: `cd src/backend && CLAUDEDJ_NO_WEBCAM=1 python -m pytest tests/ -v`
Expected: All tests across all test files PASS

- [ ] **Step 6: Commit**

```bash
git add src/backend/main.py src/backend/tests/test_api.py
git commit -m "feat: add FastAPI reaction endpoints and reactor lifecycle"
```

---

### Task 6: Add emotion_confidence to Redis store serialization

**Files:**
- Modify: `src/backend/store.py:63-88` (add emotion_confidence to frame_data)

The new `emotion_confidence` field on ReactionFrame needs to flow through to Redis storage.

- [ ] **Step 1: Update store_reaction_frame to include emotion_confidence**

In `src/backend/store.py`, in the `store_reaction_frame` function, add `emotion_confidence` to the `frame_data` dict (after the `dominant_emotion` line):

```python
    frame_data = {
        "timestamp": frame.timestamp,
        "presence": frame.presence,
        "movement": frame.movement,
        "head_pose": head_pose_data,
        "face": frame.face,
        "raw_emotions": frame.raw_emotions,
        "emotions": frame.emotions,
        "dominant_emotion": frame.dominant_emotion,
        "emotion_confidence": frame.emotion_confidence,
        "playback": frame.playback,
        "vocal": frame.vocal,
        "source": frame.source.value,
    }
```

- [ ] **Step 2: Commit**

```bash
git add src/backend/store.py
git commit -m "feat: include emotion_confidence in Redis reaction frame storage"
```

---

## Summary

| Task | What it does | Files touched |
|------|-------------|---------------|
| 1 | Fix broken test_face.py _ensemble_emotions call | test_face.py |
| 2 | Add emotion confidence (distribution entropy) to pipeline | reaction.py, webcam.py, tests/ |
| 3 | Context-conditioned interpretation for track energy (FR-7) | reaction.py, reactor.py, tests/ |
| 4 | Unit tests for Reactor CLI-only mode | tests/test_reactor.py |
| 5 | FastAPI reaction endpoints + reactor lifecycle | main.py, tests/test_api.py |
| 6 | Wire emotion_confidence to Redis storage | store.py |

After all tasks: the backend emotion detector fully supports engaged/disinterested/neutral classification with context-aware scoring, confidence weighting, proper tests, and HTTP endpoints for the agent to query.
