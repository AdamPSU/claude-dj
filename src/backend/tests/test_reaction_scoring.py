import time
import unittest

from claude_dj.reactions.models import Baseline, ReactionFrame, Sentiment
from claude_dj.reactions.scoring import aggregate_window, capture_baseline, emotion_confidence


class ReactionScoringTests(unittest.TestCase):
    def test_negative_collapsed_emotions_score_negative(self) -> None:
        frame = ReactionFrame(
            timestamp=time.time(),
            presence=1.0,
            movement=0.0,
            face=0.1,
            emotions={"happy": 0.0, "neutral": 0.2, "disinterested": 0.8},
        )
        baseline = Baseline(
            movement=0.3,
            face=0.5,
            emotions={"happy": 0.3, "neutral": 0.5, "disinterested": 0.2},
        )

        score = aggregate_window([frame], baseline)

        self.assertEqual(score.sentiment, Sentiment.NEGATIVE)
        self.assertLess(score.score, 0.4)

    def test_capture_baseline_preserves_average_collapsed_emotions(self) -> None:
        frames = [
            ReactionFrame(movement=0.1, face=0.4, emotions={"happy": 0.2, "neutral": 0.6, "disinterested": 0.2}),
            ReactionFrame(movement=0.3, face=0.6, emotions={"happy": 0.4, "neutral": 0.4, "disinterested": 0.2}),
        ]

        baseline = capture_baseline(frames)

        self.assertEqual(baseline.frame_count, 2)
        self.assertAlmostEqual(baseline.movement, 0.2)
        self.assertAlmostEqual(baseline.face, 0.5)
        self.assertAlmostEqual(baseline.emotions["happy"], 0.3)

    def test_emotion_confidence_increases_when_distribution_is_peaked(self) -> None:
        flat = emotion_confidence({"happy": 0.34, "neutral": 0.33, "disinterested": 0.33})
        peaked = emotion_confidence({"happy": 0.9, "neutral": 0.05, "disinterested": 0.05})

        self.assertLess(flat, peaked)
        self.assertGreater(peaked, 0.8)


if __name__ == "__main__":
    unittest.main()
