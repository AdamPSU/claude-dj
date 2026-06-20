"""Webcam reaction worker for ClaudeDJ.

Captures frames from the webcam at ~1fps, extracts presence, movement,
and facial expression signals, and produces ReactionFrames. Runs in a
background thread so it never blocks playback (P6).

Uses MediaPipe FaceLandmarker for face detection/presence and FER
(Facial Expression Recognition) for emotion classification.

Privacy: processes frames locally, stores only derived scores (P7).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fer.fer import FER as FERDetector

from reaction import Baseline, ReactionFrame, SignalSource, capture_baseline

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model path — sits alongside this file
_MODEL_PATH = str(Path(__file__).parent / "face_landmarker.task")

# Weights for computing engagement score from FER emotions.
# Positive emotions (happy, surprise) boost engagement;
# negative emotions (angry, sad, fear, disgust) signal disengagement;
# neutral is baseline.
_EMOTION_WEIGHTS: dict[str, float] = {
    "happy": 1.0,
    "surprise": 0.8,
    "neutral": 0.3,
    "sad": -0.4,
    "angry": -0.6,
    "fear": -0.3,
    "disgust": -0.7,
}


def _frame_difference(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Movement score from frame differencing (0.0-1.0)."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff) / 255.0)


def _fer_engagement_score(emotions: dict[str, float]) -> float:
    """Compute engagement score (0.0-1.0) from FER emotion probabilities.

    Maps the emotion vector to a single engagement score using weighted sum,
    then clamps to [0, 1]. High happy/surprise = high engagement,
    high sad/angry/disgust = low engagement.
    """
    raw = sum(emotions.get(k, 0.0) * w for k, w in _EMOTION_WEIGHTS.items())
    # raw range is roughly -0.7 to 1.0, map to 0-1
    return round(max(0.0, min(1.0, 0.5 + raw * 0.6)), 3)


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
        model_path: str = _MODEL_PATH,
        use_fer: bool = True,
    ):
        self.camera_index = camera_index
        self.sample_interval = sample_interval
        self.buffer_size = buffer_size
        self.baseline_frames = baseline_frames
        self.model_path = model_path
        self.use_fer = use_fer

        self._fer: FERDetector | None = None
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
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at {self.model_path}. "
                "Download it with: curl -L -o face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task"
            )
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

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with FaceLandmarker.create_from_options(options) as landmarker:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(self.sample_interval)
                    continue

                h, _w, _ = frame.shape
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect(mp_image)

                face_detected = len(results.face_landmarks) > 0

                presence = 1.0 if face_detected else 0.0
                face_score: float | None = None
                if face_detected:
                    landmarks = results.face_landmarks[0]
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
