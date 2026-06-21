"""Local audio playback via pygame.mixer.

Plays .mp3/.wav/.ogg files from a tracks directory. The Player class
exposes play/next/stop and current position. Structured so a Spotify
adapter can be dropped in behind the same interface.
"""

from __future__ import annotations

import os

import pygame

from vibe_dj import config


class Player:
    """Local file playback backend."""

    def __init__(self, tracks_dir: str = config.TRACKS_DIR):
        pygame.mixer.init(frequency=44100)
        self._tracks_dir = tracks_dir
        self._tracks: list[str] = []
        self._current_idx: int = -1

    def load_tracks(self, directory: str | None = None) -> list[str]:
        """Scan directory for audio files. Returns the list of track names."""
        d = directory or self._tracks_dir
        if not os.path.isdir(d):
            return []
        self._tracks = sorted([
            os.path.join(d, f) for f in os.listdir(d)
            if f.lower().endswith((".mp3", ".wav", ".ogg"))
        ])
        return [os.path.basename(t) for t in self._tracks]

    def play(self, index: int = 0) -> str:
        """Play track at index. Returns the track filename."""
        if not self._tracks:
            return ""
        self._current_idx = index % len(self._tracks)
        pygame.mixer.music.load(self._tracks[self._current_idx])
        pygame.mixer.music.play()
        return self.current_track

    def next_track(self) -> str:
        """Advance to next track. Returns the new track filename."""
        return self.play(self._current_idx + 1)

    def stop(self) -> None:
        pygame.mixer.music.stop()

    def get_position_s(self) -> float:
        """Current playback position in seconds."""
        return pygame.mixer.music.get_pos() / 1000.0

    @property
    def current_track(self) -> str:
        if 0 <= self._current_idx < len(self._tracks):
            return os.path.basename(self._tracks[self._current_idx])
        return ""

    @property
    def is_playing(self) -> bool:
        return pygame.mixer.music.get_busy()

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    @property
    def current_path(self) -> str:
        if 0 <= self._current_idx < len(self._tracks):
            return self._tracks[self._current_idx]
        return ""
