"""Tests for emotion pipeline: confidence, scoring, ensemble, smoothing."""
import pytest
from reaction import ReactionFrame, emotion_confidence

class TestEmotionConfidence:
    """emotion_confidence: distribution entropy -> 0.0-1.0 confidence."""
    def test_peaked_distribution_high_confidence(self):
        emotions = {"happy": 0.9, "neutral": 0.05, "disinterested": 0.05}
        conf = emotion_confidence(emotions)
        assert conf >= 0.8
        assert conf <= 1.0

    def test_uniform_distribution_low_confidence(self):
        emotions = {"happy": 0.333, "neutral": 0.333, "disinterested": 0.334}
        conf = emotion_confidence(emotions)
        assert conf <= 0.15

    def test_empty_distribution_zero_confidence(self):
        conf = emotion_confidence({})
        assert conf == 0.0

    def test_all_zeros_zero_confidence(self):
        emotions = {"happy": 0.0, "neutral": 0.0, "disinterested": 0.0}
        conf = emotion_confidence(emotions)
        assert conf == 0.0

    def test_moderate_distribution_moderate_confidence(self):
        emotions = {"happy": 0.6, "neutral": 0.3, "disinterested": 0.1}
        conf = emotion_confidence(emotions)
        assert 0.3 <= conf <= 0.7

    def test_returns_float_in_range(self):
        emotions = {"happy": 0.5, "neutral": 0.3, "disinterested": 0.2}
        conf = emotion_confidence(emotions)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0
