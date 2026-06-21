"""Tests for emotion classification: 3-bucket collapse, EMA, valence."""

from vibe_dj.emotion import collapse_to_buckets, ema_smooth, to_valence


class TestCollapseToBuckets:
    def test_happy_maps_to_positive(self):
        raw = {"happy": 80.0, "surprise": 10.0, "neutral": 5.0,
               "angry": 1.0, "sad": 1.0, "fear": 1.0, "disgust": 2.0}
        buckets = collapse_to_buckets(raw)
        assert buckets["positive"] > 0.8

    def test_angry_maps_to_negative(self):
        raw = {"happy": 1.0, "surprise": 1.0, "neutral": 3.0,
               "angry": 70.0, "sad": 15.0, "fear": 5.0, "disgust": 5.0}
        buckets = collapse_to_buckets(raw)
        assert buckets["negative"] > 0.8

    def test_neutral_dominant(self):
        raw = {"happy": 5.0, "surprise": 5.0, "neutral": 80.0,
               "angry": 2.0, "sad": 3.0, "fear": 2.0, "disgust": 3.0}
        buckets = collapse_to_buckets(raw)
        assert buckets["neutral"] > 0.7

    def test_buckets_sum_to_one(self):
        raw = {"happy": 20.0, "surprise": 10.0, "neutral": 30.0,
               "angry": 15.0, "sad": 10.0, "fear": 5.0, "disgust": 10.0}
        buckets = collapse_to_buckets(raw)
        assert abs(sum(buckets.values()) - 1.0) < 0.01

    def test_surprise_goes_to_positive(self):
        raw = {"happy": 5.0, "surprise": 85.0, "neutral": 5.0,
               "angry": 1.0, "sad": 1.0, "fear": 1.0, "disgust": 2.0}
        buckets = collapse_to_buckets(raw)
        assert buckets["positive"] > 0.8


class TestEmaSmooth:
    def test_first_call_returns_current(self):
        current = {"positive": 0.8, "neutral": 0.1, "negative": 0.1}
        result = ema_smooth(current, None, alpha=0.35)
        assert result == current

    def test_smoothing_moves_toward_current(self):
        prev = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
        curr = {"positive": 1.0, "neutral": 0.0, "negative": 0.0}
        result = ema_smooth(curr, prev, alpha=0.35)
        assert result["positive"] > 0.3
        assert result["positive"] < 0.4
        assert result["neutral"] > 0.6


class TestToValence:
    def test_full_positive_is_one(self):
        probs = {"positive": 1.0, "neutral": 0.0, "negative": 0.0}
        assert to_valence(probs) == 1.0

    def test_full_negative_is_zero(self):
        probs = {"positive": 0.0, "neutral": 0.0, "negative": 1.0}
        assert to_valence(probs) == 0.0

    def test_neutral_is_half(self):
        probs = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
        assert to_valence(probs) == 0.5

    def test_mixed_is_between(self):
        probs = {"positive": 0.6, "neutral": 0.2, "negative": 0.2}
        v = to_valence(probs)
        assert 0.5 < v < 1.0
