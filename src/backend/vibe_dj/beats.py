"""Beat source: interface + LibrosaGrid.

LibrosaGrid pre-extracts beats offline for reliability and exact phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BeatSource(ABC):
    """Abstract beat source exposing BPM and beat times."""

    @abstractmethod
    def load(self, audio_path: str) -> None:
        """Load/analyze a track and extract its beat grid."""
        ...

    @property
    @abstractmethod
    def bpm(self) -> float: ...

    @property
    @abstractmethod
    def all_beat_times(self) -> list[float]: ...

    @abstractmethod
    def beats_in_window(self, t0: float, t1: float) -> list[float]:
        """Return beat timestamps within [t0, t1]."""
        ...


class LibrosaGrid(BeatSource):
    """Offline beat extraction via librosa.beat.beat_track."""

    def __init__(self):
        self._bpm: float = 0.0
        self._beat_times: list[float] = []

    def load(self, audio_path: str) -> None:
        import librosa

        y, sr = librosa.load(audio_path, sr=22050)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        self._bpm = float(np.atleast_1d(tempo)[0])
        self._beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def all_beat_times(self) -> list[float]:
        return list(self._beat_times)

    def beats_in_window(self, t0: float, t1: float) -> list[float]:
        return [t for t in self._beat_times if t0 <= t <= t1]
