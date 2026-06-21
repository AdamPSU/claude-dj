import unittest

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


if __name__ == "__main__":
    unittest.main()
