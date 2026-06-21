import time
import unittest

from claude_dj.reactions.reaction import (
    Baseline,
    LandmarkExpression,
    ReactionFrame,
    Sentiment,
    TrackContext,
    TrackReaction,
    aggregate_window,
    capture_baseline,
    context_aware_collapse,
    emotion_confidence,
)


class ReactionScoringTests(unittest.TestCase):
    def test_disinterested_collapsed_emotions_score_negative(self) -> None:
        frame = ReactionFrame(
            timestamp=time.time(),
            presence=1.0,
            movement=0.0,
            face=0.1,
            emotions={"happy": 0.0, "neutral": 0.2, "disinterested": 0.8},
            emotion_confidence=0.8,
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

    def test_context_aware_collapse_treats_sad_face_during_sad_track_as_engaged(self) -> None:
        raw = {"angry": 0.05, "disgust": 0.05, "fear": 0.05, "happy": 0.05, "sad": 0.7, "surprise": 0.05, "neutral": 0.05}

        collapsed = context_aware_collapse(raw, TrackContext(energy=0.2, valence=0.2))

        self.assertGreater(collapsed["happy"], collapsed["disinterested"])
        self.assertAlmostEqual(sum(collapsed.values()), 1.0, places=3)

    def test_landmark_smile_can_drive_positive_score_without_cnn_emotion(self) -> None:
        frame = ReactionFrame(landmark_expression=LandmarkExpression(smile=0.7, mouth_open=0.1, ear=0.3))
        baseline = Baseline(landmark_smile=0.1, landmark_mouth=0.05, landmark_ear=0.3)

        score = aggregate_window([frame], baseline)

        self.assertEqual(score.sentiment, Sentiment.POSITIVE)
        self.assertGreater(score.score, 0.6)

    def test_track_reaction_records_full_reaction_trace(self) -> None:
        trace = TrackReaction(track_id="track-1", frames=[ReactionFrame()], scores=[])

        self.assertEqual(trace.track_id, "track-1")
        self.assertEqual(len(trace.frames), 1)
        self.assertIsNone(trace.final_sentiment)


if __name__ == "__main__":
    unittest.main()
