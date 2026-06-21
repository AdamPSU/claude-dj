"""Thread-safe shared state for the vibe_dj system."""

from __future__ import annotations

import threading
import time
from collections import deque


class SystemState:
    """Central state object shared between video, audio, and main threads.

    Every public method acquires the lock. Callers never touch _lock directly.
    """

    def __init__(self, pitch_buffer_maxlen: int = 300):
        self._lock = threading.Lock()

        # Head pose (written by video thread)
        self.pitch: float = 0.0
        self.yaw: float = 0.0
        self.roll: float = 0.0
        self.face_scale: float = 1.0
        self.face_detected: bool = False

        # Raw pitch buffer — UNSMOOTHED (for vibe DSP)
        self._pitch_buffer: deque[tuple[float, float]] = deque(maxlen=pitch_buffer_maxlen)

        # Face crop for emotion classifier (copy-on-write)
        self._face_crop = None  # numpy ndarray or None

        # Emotion (written by video thread at lower cadence)
        self.valence: float = 0.5
        self.emotion_bucket: str = "neutral"
        self.emotion_probs: dict[str, float] = {
            "positive": 0.0, "neutral": 1.0, "negative": 0.0,
        }

        # Vibe (written by main loop)
        self.vibe_score: float = 0.0
        self.plv: float = 0.0
        self.period_match_score: float = 0.0

        # Motion energy — magnitude of frame-to-frame pose change (0..1)
        self.motion_energy: float = 0.0

        # Audio / beats (written by audio thread)
        self.bpm: float = 120.0
        self.beat_times: list[float] = []
        self.playback_time: float = 0.0
        self.current_track: str = ""
        self.is_playing: bool = False

        # Agent (written by main loop)
        self.agent_action: str = ""
        self.agent_reason: str = ""

    # --- Pose updates (video thread) ---

    def update_pose(
        self, pitch: float, yaw: float, roll: float,
        face_scale: float, face_detected: bool,
    ) -> None:
        with self._lock:
            self.pitch = pitch
            self.yaw = yaw
            self.roll = roll
            self.face_scale = face_scale
            self.face_detected = face_detected

    def append_pitch(self, timestamp: float, pitch: float) -> None:
        with self._lock:
            self._pitch_buffer.append((timestamp, pitch))

    def get_pitch_window(self, window_s: float) -> list[tuple[float, float]]:
        with self._lock:
            now = time.time()
            cutoff = now - window_s
            return [(t, p) for t, p in self._pitch_buffer if t >= cutoff]

    # --- Face crop (video thread writes, emotion reads) ---

    def set_face_crop(self, crop) -> None:
        with self._lock:
            self._face_crop = crop.copy() if crop is not None else None

    def get_face_crop(self):
        with self._lock:
            return self._face_crop.copy() if self._face_crop is not None else None

    # --- Emotion (video thread) ---

    def update_emotion(
        self, valence: float, bucket: str, probs: dict[str, float],
    ) -> None:
        with self._lock:
            self.valence = valence
            self.emotion_bucket = bucket
            self.emotion_probs = probs.copy()

    # --- Vibe (main loop) ---

    def update_vibe(
        self, vibe_score: float, plv: float, period_match: float,
    ) -> None:
        with self._lock:
            self.vibe_score = vibe_score
            self.plv = plv
            self.period_match_score = period_match

    # --- Beats (audio thread) ---

    def update_beats(self, bpm: float, beat_times: list[float]) -> None:
        with self._lock:
            self.bpm = bpm
            self.beat_times = list(beat_times)

    def update_playback(self, position: float, track: str, playing: bool) -> None:
        with self._lock:
            self.playback_time = position
            self.current_track = track
            self.is_playing = playing

    # --- Agent (main loop) ---

    def update_agent(self, action: str, reason: str) -> None:
        with self._lock:
            self.agent_action = action
            self.agent_reason = reason

    # --- Snapshot (main loop reads for HUD / agent) ---

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pitch": self.pitch,
                "yaw": self.yaw,
                "roll": self.roll,
                "face_detected": self.face_detected,
                "face_scale": self.face_scale,
                "valence": self.valence,
                "emotion_bucket": self.emotion_bucket,
                "emotion_probs": dict(self.emotion_probs),
                "vibe_score": self.vibe_score,
                "plv": self.plv,
                "period_match_score": self.period_match_score,
                "motion_energy": self.motion_energy,
                "bpm": self.bpm,
                "playback_time": self.playback_time,
                "current_track": self.current_track,
                "is_playing": self.is_playing,
                "agent_action": self.agent_action,
                "agent_reason": self.agent_reason,
            }
