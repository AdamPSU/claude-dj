"""Tests for context-aware emotion collapse.

Validates that raw 7-class emotions are collapsed differently depending on
track context (energy, valence). Sad face during a sad song = engaged,
sad face during a party song = disengaged.
"""
import pytest


# A reusable raw emotion distribution for testing.
# Each class gets a clear share so we can verify where it lands.
def _raw(dominant: str, score: float = 0.7) -> dict[str, float]:
    """Build a raw 7-class distribution peaked at `dominant`."""
    keys = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    remainder = (1.0 - score) / (len(keys) - 1)
    return {k: (score if k == dominant else remainder) for k in keys}


class TestTrackContextHasValence:
    """TrackContext must have a valence field for context-aware collapse."""

    def test_valence_field_exists(self):
        from reaction import TrackContext
        ctx = TrackContext(energy=0.5, valence=0.7)
        assert ctx.valence == 0.7

    def test_valence_defaults(self):
        from reaction import TrackContext
        ctx = TrackContext()
        assert hasattr(ctx, "valence")


class TestContextAwareCollapseNoContext:
    """Without track context, behaves identically to the static mapping."""

    def test_happy_maps_to_happy(self):
        from reaction import context_aware_collapse
        result = context_aware_collapse(_raw("happy"))
        assert result["happy"] > result["disinterested"]

    def test_sad_maps_to_disinterested(self):
        from reaction import context_aware_collapse
        result = context_aware_collapse(_raw("sad"))
        assert result["disinterested"] > result["happy"]

    def test_neutral_maps_to_neutral(self):
        from reaction import context_aware_collapse
        result = context_aware_collapse(_raw("neutral"))
        assert result["neutral"] > result["happy"]
        assert result["neutral"] > result["disinterested"]

    def test_output_sums_to_one(self):
        from reaction import context_aware_collapse
        result = context_aware_collapse(_raw("happy"))
        assert abs(sum(result.values()) - 1.0) < 0.01


class TestContextCollapsePartyTrack:
    """High energy + high valence (party music). Same as static mapping."""

    def test_happy_stays_happy(self):
        from reaction import context_aware_collapse, TrackContext
        ctx = TrackContext(energy=0.8, valence=0.8)
        result = context_aware_collapse(_raw("happy"), ctx)
        assert result["happy"] > result["disinterested"]

    def test_sad_stays_disinterested(self):
        from reaction import context_aware_collapse, TrackContext
        ctx = TrackContext(energy=0.8, valence=0.8)
        result = context_aware_collapse(_raw("sad"), ctx)
        assert result["disinterested"] > result["happy"]


class TestContextCollapseSadBallad:
    """Low energy + low valence (sad ballad). Sadness = matching the vibe."""

    def test_sad_becomes_engaged(self):
        from reaction import context_aware_collapse, TrackContext
        ctx = TrackContext(energy=0.2, valence=0.2)
        result = context_aware_collapse(_raw("sad"), ctx)
        # Sad during a sad song is engagement, not disinterest
        assert result["happy"] > result["disinterested"], \
            f"Sad face during sad song should read as engaged: {result}"

    def test_happy_still_engaged(self):
        from reaction import context_aware_collapse, TrackContext
        ctx = TrackContext(energy=0.2, valence=0.2)
        result = context_aware_collapse(_raw("happy"), ctx)
        assert result["happy"] > result["disinterested"]

    def test_disgust_still_disinterested(self):
        from reaction import context_aware_collapse, TrackContext
        ctx = TrackContext(energy=0.2, valence=0.2)
        result = context_aware_collapse(_raw("disgust"), ctx)
        assert result["disinterested"] > result["happy"]


class TestContextCollapseIntenseTrack:
    """Low valence + high energy (metal, intense EDM)."""

    def test_angry_becomes_engaged(self):
        from reaction import context_aware_collapse, TrackContext
        ctx = TrackContext(energy=0.9, valence=0.3)
        result = context_aware_collapse(_raw("angry"), ctx)
        assert result["happy"] > result["disinterested"], \
            f"Angry face during intense track should read as engaged: {result}"

    def test_fear_becomes_engaged(self):
        from reaction import context_aware_collapse, TrackContext
        ctx = TrackContext(energy=0.8, valence=0.3)
        result = context_aware_collapse(_raw("fear"), ctx)
        assert result["happy"] > result["disinterested"], \
            f"Fear/intensity during high-energy track should read as engaged: {result}"

    def test_disgust_still_disinterested(self):
        from reaction import context_aware_collapse, TrackContext
        ctx = TrackContext(energy=0.9, valence=0.3)
        result = context_aware_collapse(_raw("disgust"), ctx)
        assert result["disinterested"] > result["happy"]


class TestContextCollapseInvariant:
    """Properties that hold regardless of context."""

    def test_disgust_always_disinterested(self):
        from reaction import context_aware_collapse, TrackContext
        for e, v in [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.9, 0.1)]:
            ctx = TrackContext(energy=e, valence=v)
            result = context_aware_collapse(_raw("disgust"), ctx)
            assert result["disinterested"] > result["happy"], \
                f"Disgust should always be disinterested at energy={e} valence={v}"

    def test_happy_always_engaged(self):
        from reaction import context_aware_collapse, TrackContext
        for e, v in [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.1, 0.9)]:
            ctx = TrackContext(energy=e, valence=v)
            result = context_aware_collapse(_raw("happy"), ctx)
            assert result["happy"] > result["disinterested"], \
                f"Happy should always be engaged at energy={e} valence={v}"

    def test_surprise_always_engaged(self):
        from reaction import context_aware_collapse, TrackContext
        for e, v in [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]:
            ctx = TrackContext(energy=e, valence=v)
            result = context_aware_collapse(_raw("surprise"), ctx)
            assert result["happy"] > result["disinterested"]

    def test_output_always_sums_to_one(self):
        from reaction import context_aware_collapse, TrackContext
        for dominant in ["happy", "sad", "angry", "fear", "disgust", "neutral", "surprise"]:
            for e, v in [(0.2, 0.2), (0.8, 0.8), (0.9, 0.2)]:
                ctx = TrackContext(energy=e, valence=v)
                result = context_aware_collapse(_raw(dominant), ctx)
                assert abs(sum(result.values()) - 1.0) < 0.01, \
                    f"Should sum to 1.0 for {dominant} at e={e} v={v}: {result}"


class TestAggregateWindowUsesContextCollapse:
    """aggregate_window should use context-aware collapse when raw_emotions available."""

    def test_sad_face_scores_higher_during_sad_track(self):
        from reaction import (
            ReactionFrame, Baseline, TrackContext, aggregate_window,
        )
        # Sad face frame with raw 7-class emotions
        sad_raw = _raw("sad", 0.7)
        sad_collapsed_static = {"happy": 0.05, "neutral": 0.05, "disinterested": 0.9}
        frame = ReactionFrame(
            face=0.1,
            raw_emotions=sad_raw,
            emotions=sad_collapsed_static,
            dominant_emotion="sad",
            emotion_confidence=0.8,
        )
        baseline = Baseline()

        # Without context: sad = disinterested = low score
        score_no_ctx = aggregate_window([frame], baseline)

        # With sad-track context: sad = engaged = higher score
        sad_ctx = TrackContext(energy=0.2, valence=0.2)
        score_sad_ctx = aggregate_window([frame], baseline, track_context=sad_ctx)

        assert score_sad_ctx.score > score_no_ctx.score, \
            f"Sad face during sad track should score higher: ctx={score_sad_ctx.score} vs no_ctx={score_no_ctx.score}"
