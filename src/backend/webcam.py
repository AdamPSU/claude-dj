"""Webcam reaction worker for ClaudeDJ.

Captures frames from the webcam at ~1fps, extracts presence, movement,
and facial expression signals, and produces ReactionFrames. Runs in a
background thread so it never blocks playback (P6).

Privacy: processes frames locally, stores only derived scores (P7).
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from reaction import Baseline, ReactionFrame, SignalSource, capture_baseline

# MediaPipe face mesh for expression deltas
_mp_face_mesh = mp.solutions.face_mesh

# Landmark indices for expression scoring
# Mouth openness: top lip (13) vs bottom lip (14)
_UPPER_LIP = 13
_LOWER_LIP = 14
# Eyebrow raise: brow (70) vs eye corner (33)
_LEFT_BROW = 70
_LEFT_EYE_CORNER = 33


def _frame_difference(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Movement score from frame differencing (0.0–1.0)."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff) / 255.0)


def _expression_score(landmarks: list, frame_h: int) -> float:
    """Simple expression engagement score from face mesh landmarks (0.0–1.0).

    Combines mouth openness and eyebrow raise as a proxy for engagement.
    Higher = more expressive (smiling, singing, reacting).
    """
    # Mouth openness
    upper = landmarks[_UPPER_LIP]
    lower = landmarks[_LOWER_LIP]
    mouth_gap = abs(lower.y - upper.y) * frame_h

    # Eyebrow raise
    brow = landmarks[_LEFT_BROW]
    eye = landmarks[_LEFT_EYE_CORNER]
    brow_raise = abs(brow.y - eye.y) * frame_h

    # Normalize to rough 0–1 range (tuned for typical webcam distances)
    mouth_score = min(1.0, mouth_gap / 30.0)
    brow_score = min(1.0, brow_raise / 40.0)

    return round((mouth_score * 0.6 + brow_score * 0.4), 3)


class WebcamWorker:
    """Background webcam reaction capture.

    Usage:
        worker = WebcamWorker()
        worker.start()
        # ... later ...
        frames = worker.get_recent_frames(n=10)
        worker.stop()
    """

    def __init__(
        self,
        camera_index: int = 0,
        sample_interval: float = 1.0,
        buffer_size: int = 120,
        baseline_frames: int = 3,
    ):
        self.camera_index = camera_index
        self.sample_interval = sample_interval
        self.buffer_size = buffer_size
        self.baseline_frames = baseline_frames

        self._frames: deque[ReactionFrame] = deque(maxlen=buffer_size)
        self._baseline: Baseline | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def baseline(self) -> Baseline | None:
        return self._baseline

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_recent_frames(self, n: int = 10) -> list[ReactionFrame]:
        with self._lock:
            return list(self._frames)[-n:]

    def get_all_frames(self) -> list[ReactionFrame]:
        with self._lock:
            return list(self._frames)

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self._running = False
            return

        prev_gray: np.ndarray | None = None
        baseline_buffer: list[ReactionFrame] = []

        with _mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(self.sample_interval)
                    continue

                h, _w, _ = frame.shape
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Presence + expression from MediaPipe
                results = face_mesh.process(rgb)
                face_detected = (
                    results.multi_face_landmarks is not None
                    and len(results.multi_face_landmarks) > 0
                )

                presence = 1.0 if face_detected else 0.0
                face_score: float | None = None
                if face_detected:
                    landmarks = results.multi_face_landmarks[0].landmark
                    face_score = _expression_score(landmarks, h)

                # Movement from frame differencing
                movement: float | None = None
                if prev_gray is not None:
                    movement = _frame_difference(prev_gray, gray)
                prev_gray = gray

                reaction_frame = ReactionFrame(
                    timestamp=time.time(),
                    presence=presence,
                    movement=movement,
                    face=face_score,
                    source=SignalSource.WEBCAM,
                )

                with self._lock:
                    self._frames.append(reaction_frame)

                # Capture baseline from first N frames
                if self._baseline is None:
                    baseline_buffer.append(reaction_frame)
                    if len(baseline_buffer) >= self.baseline_frames:
                        self._baseline = capture_baseline(baseline_buffer)

                time.sleep(self.sample_interval)

        cap.release()
