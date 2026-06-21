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
        assert score.confidence == 1.0

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
