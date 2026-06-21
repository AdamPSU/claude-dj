import unittest

from claude_dj.reactions.webcam import collapse_raw_emotions, engagement_score, smooth_emotions


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

        raw_normalized, collapsed = collapse_raw_emotions(raw)

        self.assertAlmostEqual(sum(raw_normalized.values()), 1.0, places=3)
        self.assertGreater(collapsed["happy"], 0.85)
        self.assertLess(collapsed["disinterested"], 0.1)

    def test_smoothing_moves_toward_current_distribution(self) -> None:
        previous = {"happy": 0.0, "neutral": 1.0, "disinterested": 0.0}
        current = {"happy": 1.0, "neutral": 0.0, "disinterested": 0.0}

        smoothed = smooth_emotions(current, previous, alpha=0.35)

        self.assertGreater(smoothed["happy"], 0.3)
        self.assertLess(smoothed["happy"], 0.4)
        self.assertGreater(smoothed["neutral"], 0.6)

    def test_engagement_score_uses_happy_probability(self) -> None:
        self.assertEqual(engagement_score({"happy": 0.83, "neutral": 0.1, "disinterested": 0.07}), 0.83)


if __name__ == "__main__":
    unittest.main()
