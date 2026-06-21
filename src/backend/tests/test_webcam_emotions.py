import unittest

from claude_dj.reactions.emotion import collapse_to_buckets, ema_smooth, to_valence


class WebcamEmotionTests(unittest.TestCase):
    def test_raw_deepface_emotions_collapse_to_three_states(self) -> None:
        raw = {
            "happy": 80.0,
            "surprise": 10.0,
            "neutral": 5.0,
            "sad": 2.0,
            "angry": 1.0,
            "fear": 1.0,
            "disgust": 1.0,
        }

        buckets = collapse_to_buckets(raw)

        self.assertAlmostEqual(sum(buckets.values()), 1.0, places=3)
        self.assertGreater(buckets["positive"], 0.85)
        self.assertLess(buckets["negative"], 0.1)

    def test_smoothing_moves_toward_current_distribution(self) -> None:
        previous = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
        current = {"positive": 1.0, "neutral": 0.0, "negative": 0.0}

        smoothed = ema_smooth(current, previous, alpha=0.35)

        self.assertGreater(smoothed["positive"], 0.3)
        self.assertLess(smoothed["positive"], 0.4)
        self.assertGreater(smoothed["neutral"], 0.6)

    def test_valence_maps_buckets_to_zero_to_one_score(self) -> None:
        self.assertEqual(to_valence({"positive": 1.0, "neutral": 0.0, "negative": 0.0}), 1.0)
        self.assertEqual(to_valence({"positive": 0.0, "neutral": 0.0, "negative": 1.0}), 0.0)
        self.assertEqual(to_valence({"positive": 0.0, "neutral": 1.0, "negative": 0.0}), 0.5)


if __name__ == "__main__":
    unittest.main()
