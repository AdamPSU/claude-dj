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
    def test_high_energy_track_boosts_movement(self):
        """Movement during up-tempo track should boost engagement."""
        frames = [_make_frame(movement=0.6, face=0.5,
                              emotions={"happy": 0.4, "neutral": 0.4, "disinterested": 0.2})]
        baseline = Baseline(movement=0.1, face=0.5,
                            emotions={"happy": 0.4, "neutral": 0.4, "disinterested": 0.2})
        ctx = TrackContext(energy=0.9)
        score_with_ctx = aggregate_window(frames, baseline, track_context=ctx)
        score_without = aggregate_window(frames, baseline)
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
