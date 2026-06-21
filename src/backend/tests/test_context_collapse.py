import unittest

from claude_dj.reactions.reaction import Baseline, ReactionFrame, TrackContext, aggregate_window, context_aware_collapse


def raw(dominant: str, score: float = 0.7) -> dict[str, float]:
    keys = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    remainder = (1.0 - score) / (len(keys) - 1)
    return {key: (score if key == dominant else remainder) for key in keys}


class ContextCollapseTests(unittest.TestCase):
    def test_track_context_has_valence(self) -> None:
        context = TrackContext(energy=0.5, valence=0.7)

        self.assertEqual(context.valence, 0.7)

    def test_sad_face_without_context_maps_to_disinterested(self) -> None:
        collapsed = context_aware_collapse(raw("sad"))

        self.assertGreater(collapsed["disinterested"], collapsed["happy"])

    def test_sad_face_during_sad_track_maps_to_engaged(self) -> None:
        collapsed = context_aware_collapse(raw("sad"), TrackContext(energy=0.2, valence=0.2))

        self.assertGreater(collapsed["happy"], collapsed["disinterested"])

    def test_angry_face_during_intense_track_maps_to_engaged(self) -> None:
        collapsed = context_aware_collapse(raw("angry"), TrackContext(energy=0.9, valence=0.3))

        self.assertGreater(collapsed["happy"], collapsed["disinterested"])

    def test_disgust_always_maps_to_disinterested(self) -> None:
        for energy, valence in [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.9, 0.1)]:
            collapsed = context_aware_collapse(raw("disgust"), TrackContext(energy=energy, valence=valence))

            self.assertGreater(collapsed["disinterested"], collapsed["happy"])

    def test_aggregate_window_uses_context_collapse_when_raw_emotions_are_available(self) -> None:
        frame = ReactionFrame(
            face=0.1,
            raw_emotions=raw("sad", 0.7),
            emotions={"happy": 0.05, "neutral": 0.05, "disinterested": 0.9},
            dominant_emotion="sad",
            emotion_confidence=0.8,
        )
        baseline = Baseline()

        score_without_context = aggregate_window([frame], baseline)
        score_with_sad_context = aggregate_window([frame], baseline, track_context=TrackContext(energy=0.2, valence=0.2))

        self.assertGreater(score_with_sad_context.score, score_without_context.score)


if __name__ == "__main__":
    unittest.main()
