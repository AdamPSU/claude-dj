import unittest

from claude_dj.reactions.reaction import context_aware_collapse
from claude_dj.reactions.webcam import _deepface_to_emotions, _smooth_emotions


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

        _, buckets = _deepface_to_emotions(raw)

        self.assertAlmostEqual(sum(buckets.values()), 1.0, places=3)
        self.assertGreater(buckets["happy"], 0.85)
        self.assertLess(buckets["disinterested"], 0.1)

    def test_smoothing_moves_toward_current_distribution(self) -> None:
        previous = {"happy": 0.0, "neutral": 1.0, "disinterested": 0.0}
        current = {"happy": 1.0, "neutral": 0.0, "disinterested": 0.0}

        smoothed = _smooth_emotions(current, previous, alpha=0.35, confidence=0.0)

        self.assertGreater(smoothed["happy"], 0.3)
        self.assertLess(smoothed["happy"], 0.4)
        self.assertGreater(smoothed["neutral"], 0.6)

    def test_static_collapse_treats_disgust_as_disinterested(self) -> None:
        collapsed = context_aware_collapse({"disgust": 0.8, "neutral": 0.2})

        self.assertGreater(collapsed["disinterested"], collapsed["happy"])
        self.assertAlmostEqual(sum(collapsed.values()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
