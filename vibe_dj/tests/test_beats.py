"""Tests for beat source interface and LibrosaGrid."""

import numpy as np
import pytest
from vibe_dj.beats import LibrosaGrid


class TestLibrosaGrid:
    @pytest.fixture
    def click_track(self, tmp_path):
        """Generate a short WAV with audible clicks at 120 BPM."""
        import soundfile as sf
        sr = 22050
        duration = 4.0
        bpm = 120.0
        beat_interval = 60.0 / bpm  # 0.5s
        samples = np.zeros(int(sr * duration))
        for i in range(int(duration / beat_interval)):
            idx = int(i * beat_interval * sr)
            click_len = min(200, len(samples) - idx)
            samples[idx: idx + click_len] = 0.8 * np.sin(
                2 * np.pi * 1000 * np.arange(click_len) / sr
            )
        path = tmp_path / "click_120bpm.wav"
        sf.write(str(path), samples, sr)
        return str(path)

    def test_load_sets_bpm(self, click_track):
        grid = LibrosaGrid()
        grid.load(click_track)
        assert 90 < grid.bpm < 150

    def test_beat_times_are_sorted(self, click_track):
        grid = LibrosaGrid()
        grid.load(click_track)
        times = grid.all_beat_times
        assert times == sorted(times)

    def test_beats_in_window(self, click_track):
        grid = LibrosaGrid()
        grid.load(click_track)
        beats = grid.beats_in_window(0.0, 2.0)
        assert len(beats) >= 2

    def test_beats_in_window_empty_outside_range(self, click_track):
        grid = LibrosaGrid()
        grid.load(click_track)
        beats = grid.beats_in_window(100.0, 200.0)
        assert beats == []
