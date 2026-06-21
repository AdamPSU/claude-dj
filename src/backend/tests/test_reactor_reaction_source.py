import unittest

from claude_dj.reactions.reaction import HeadPose, LandmarkExpression, ReactionFrame
from claude_dj.reactions.reactor import Reactor
from claude_dj.reactions.reactor import ReactorReactionSource


class FakeReactor:
    def __init__(self, summary: dict[str, object]) -> None:
        self.summary = summary

    def get_summary(self) -> dict[str, object]:
        return dict(self.summary)


class ReactorReactionSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_maps_reactor_summary_to_mcp_reaction_signal(self) -> None:
        source = ReactorReactionSource(
            FakeReactor(
                {
                    "current_score": 0.2,
                    "confidence": 0.91,
                    "sentiment": "negative",
                    "trend_direction": "falling",
                    "trend_scores": [0.52, 0.31, 0.2],
                    "source": "webcam",
                    "dominant_emotion": "angry",
                }
            )
        )

        signal = await source.get_reaction_signal()

        self.assertTrue(signal["available"])
        self.assertFalse(signal["stub"])
        self.assertEqual(signal["source"], "webcam")
        self.assertEqual(signal["trend"], "negative")
        self.assertAlmostEqual(signal["score"], -0.6)
        self.assertEqual(signal["confidence"], 0.91)
        self.assertEqual(signal["camera_state"], "falling")
        self.assertIn("negative", signal["summary"])
        self.assertEqual(signal["raw"]["dominant_emotion"], "angry")

    async def test_zero_confidence_summary_is_unavailable(self) -> None:
        source = ReactorReactionSource(
            FakeReactor(
                {
                    "current_score": 0.5,
                    "confidence": 0.0,
                    "sentiment": "neutral",
                    "trend_direction": "stable",
                    "source": "webcam",
                }
            )
        )

        signal = await source.get_reaction_signal()

        self.assertFalse(signal["available"])
        self.assertEqual(signal["trend"], "neutral")

    async def test_reactor_summary_includes_latest_landmark_fields(self) -> None:
        class FrameSource:
            baseline = None
            error = None

            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

            def get_recent_frames(self, n: int = 10) -> list[ReactionFrame]:
                return [
                    ReactionFrame(
                        presence=1.0,
                        movement=0.4,
                        face=0.8,
                        raw_emotions={"happy": 0.7, "surprise": 0.1, "neutral": 0.2},
                        emotions={"happy": 0.8, "neutral": 0.1, "disinterested": 0.1},
                        dominant_emotion="happy",
                        landmark_expression=LandmarkExpression(
                            smile=0.72,
                            mouth_open=0.18,
                            ear=0.31,
                            brow_height=0.56,
                        ),
                        head_pose=HeadPose(yaw=3.0, pitch=-2.0, roll=1.0),
                    )
                ]

            def get_all_frames(self) -> list[ReactionFrame]:
                return self.get_recent_frames()

        summary = Reactor(FrameSource()).get_summary()

        self.assertEqual(summary["dominant_emotion"], "happy")
        self.assertEqual(summary["emotions"], {"happy": 0.8, "neutral": 0.1, "disinterested": 0.1})
        self.assertEqual(summary["raw_emotions"], {"happy": 0.7, "surprise": 0.1, "neutral": 0.2})
        self.assertEqual(summary["landmark_expression"]["smile"], 0.72)
        self.assertEqual(summary["head_pose"]["yaw"], 3.0)


if __name__ == "__main__":
    unittest.main()
