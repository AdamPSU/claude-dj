import math
import unittest

from claude_dj.reactions.vibe import VibeDetector


def make_pitch_window(freq_hz: float, *, duration_seconds: float = 3.0, fps: int = 30) -> list[tuple[float, float]]:
    samples = int(duration_seconds * fps)
    return [
        (index / fps, 5.0 * math.sin(2.0 * math.pi * freq_hz * (index / fps)))
        for index in range(samples)
    ]


def beat_times(bpm: float, *, duration_seconds: float = 3.0) -> list[float]:
    interval = 60.0 / bpm
    count = int(duration_seconds / interval)
    return [index * interval for index in range(count)]


class VibeDetectorTests(unittest.TestCase):
    def test_on_beat_head_bobbing_scores_high(self) -> None:
        bpm = 120.0
        vibe, plv, period_match = VibeDetector().compute(make_pitch_window(bpm / 60.0), beat_times(bpm), bpm)

        self.assertGreater(vibe, 0.7)
        self.assertGreater(plv, 0.6)
        self.assertGreater(period_match, 0.6)

    def test_still_face_scores_low(self) -> None:
        pitch_window = [(index / 30.0, 0.05) for index in range(90)]

        vibe, _, _ = VibeDetector().compute(pitch_window, beat_times(120.0), 120.0)

        self.assertLess(vibe, 0.4)


if __name__ == "__main__":
    unittest.main()
