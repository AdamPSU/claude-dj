"""Tests for vibe detection DSP pipeline.

Uses synthetic pitch signals to validate behavioral properties:
- Bobbing on beat -> high vibe (>0.7)
- Still face -> low vibe (~0)
- Off-beat bobbing -> low vibe
- Half-time bobbing -> period match still works
"""

import numpy as np
import pytest
from vibe_dj.vibe import VibeDetector


@pytest.fixture
def detector():
    return VibeDetector()


def _make_pitch_window(freq_hz, duration_s=3.0, fps=30, amplitude=5.0):
    """Generate a sinusoidal pitch signal at the given frequency."""
    n = int(fps * duration_s)
    t = np.linspace(0, duration_s, n)
    pitch = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return list(zip(t.tolist(), pitch.tolist()))


def _beat_times(bpm, duration_s=3.0):
    """Generate beat timestamps."""
    interval = 60.0 / bpm
    return np.arange(0, duration_s, interval).tolist()


class TestOnBeatBobbing:
    def test_exact_beat_frequency_high_vibe(self, detector):
        """Bobbing at exactly the beat frequency -> vibe > 0.7."""
        bpm = 120
        freq = bpm / 60.0  # 2 Hz
        pw = _make_pitch_window(freq)
        beats = _beat_times(bpm)
        vibe, plv, pm = detector.compute(pw, beats, bpm)
        assert vibe > 0.7, f"On-beat bob: vibe={vibe:.3f} (expected >0.7)"
        assert plv > 0.6, f"PLV={plv:.3f}"

    def test_different_bpm(self, detector):
        """Works at 100 BPM too."""
        bpm = 100
        freq = bpm / 60.0
        pw = _make_pitch_window(freq)
        beats = _beat_times(bpm)
        vibe, _, _ = detector.compute(pw, beats, bpm)
        assert vibe > 0.6


class TestStillFace:
    def test_no_movement_low_vibe(self, detector):
        """Flat pitch (+ tiny noise) -> vibe well below on-beat threshold."""
        bpm = 120
        n = 90
        rng = np.random.default_rng(42)  # fixed seed for reproducibility
        t = np.linspace(0, 3.0, n)
        pitch = rng.normal(0, 0.1, n)
        pw = list(zip(t.tolist(), pitch.tolist()))
        beats = _beat_times(bpm)
        vibe, _, _ = detector.compute(pw, beats, bpm)
        assert vibe < 0.4, f"Still face: vibe={vibe:.3f} (expected <0.4)"


class TestOffBeatBobbing:
    def test_wrong_frequency_low_vibe(self, detector):
        """Bobbing at 1.7x the beat frequency -> low vibe."""
        bpm = 120
        beat_freq = bpm / 60.0
        off_freq = beat_freq * 1.7
        pw = _make_pitch_window(off_freq)
        beats = _beat_times(bpm)
        vibe, _, _ = detector.compute(pw, beats, bpm)
        assert vibe < 0.5, f"Off-beat: vibe={vibe:.3f} (expected <0.5)"


class TestHalfAndDoubleTime:
    def test_half_time_period_match(self, detector):
        """Bobbing at half the beat freq -> period match still works."""
        bpm = 120
        beat_freq = bpm / 60.0
        half_freq = beat_freq / 2.0  # 1 Hz
        pw = _make_pitch_window(half_freq, duration_s=4.0)
        beats = _beat_times(bpm, duration_s=4.0)
        _, _, pm = detector.compute(pw, beats, bpm)
        assert pm > 0.5, f"Half-time: period_match={pm:.3f} (expected >0.5)"

    def test_double_time_period_match(self, detector):
        """Bobbing at double the beat freq -> period match works for lower BPM."""
        bpm = 80
        beat_freq = bpm / 60.0  # ~1.33 Hz
        double_freq = beat_freq * 2.0  # ~2.67 Hz (in band)
        pw = _make_pitch_window(double_freq, duration_s=4.0)
        beats = _beat_times(bpm, duration_s=4.0)
        _, _, pm = detector.compute(pw, beats, bpm)
        assert pm > 0.4, f"Double-time: period_match={pm:.3f} (expected >0.4)"


class TestEdgeCases:
    def test_too_few_samples_returns_zero(self, detector):
        pw = [(0.0, 1.0), (0.1, 2.0)]
        vibe, _, _ = detector.compute(pw, [0.0, 0.5], 120)
        assert vibe == 0.0

    def test_no_beats_returns_zero(self, detector):
        pw = _make_pitch_window(2.0)
        vibe, _, _ = detector.compute(pw, [], 120)
        assert vibe == 0.0

    def test_zero_bpm_returns_zero(self, detector):
        pw = _make_pitch_window(2.0)
        vibe, _, _ = detector.compute(pw, [0.0, 0.5], 0)
        assert vibe == 0.0
