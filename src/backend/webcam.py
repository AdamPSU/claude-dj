"""Webcam reaction worker for ClaudeDJ.

Captures frames from the webcam at ~1fps, extracts presence, movement,
and facial expression signals, and produces ReactionFrames. Runs in a
background thread so it never blocks playback (P6).

Uses MediaPipe FaceLandmarker for face detection/presence and an ensemble
of ViT-FER (HardlyHumans/Facial-expression-detection, 92.2% accuracy)
+ DeepFace for emotion classification, with temporal smoothing
to reduce frame-to-frame noise.

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
from deepface import DeepFace
from PIL import Image
from transformers import pipeline as hf_pipeline

from reaction import Baseline, ReactionFrame, SignalSource, capture_baseline

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model path — sits alongside this file
_MODEL_PATH = str(Path(__file__).parent / "face_landmarker.task")

# ViT-FER model from HuggingFace (92.2% accuracy on FER2013+AffectNet)
_VIT_MODEL = "HardlyHumans/Facial-expression-detection"

# Canonical emotion keys (shared by ViT-FER and DeepFace)
_EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Weights for computing engagement score from emotion probabilities (0-1 scale).
_EMOTION_WEIGHTS: dict[str, float] = {
    "happy": 1.0,
    "surprise": 0.8,
    "neutral": 0.3,
    "sad": -0.4,
    "angry": -0.6,
    "fear": -0.3,
    "disgust": -0.7,
}

# Ensemble blend: ViT gets more weight due to higher accuracy.
_VIT_WEIGHT = 0.65
_DEEPFACE_WEIGHT = 0.35

# Temporal smoothing factor (exponential moving average).
# Lower = smoother but slower to react. 0.35 gives ~3-frame lag.
_SMOOTHING_ALPHA = 0.35


def _frame_difference(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Movement score from frame differencing (0.0-1.0)."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff) / 255.0)


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE to normalize lighting before emotion classification."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _vit_results_to_emotions(vit_results: list[dict]) -> dict[str, float]:
    """Convert HuggingFace pipeline output to normalized emotion dict (0-1)."""
    emos = {k: 0.0 for k in _EMOTION_KEYS}
    for item in vit_results:
        label = item["label"].lower()
        if label in emos:
            emos[label] = item["score"]
    return emos


def _ensemble_emotions(
    vit_emos: dict[str, float] | None,
    df_emos: dict[str, float] | None,
) -> dict[str, float] | None:
    """Blend ViT-FER and DeepFace emotion scores into a single distribution.

    ViT outputs 0-1 (softmax), DeepFace outputs 0-100. Both are normalized
    to 0-1 then weighted-averaged. Result sums to ~1.0.
    """
    if vit_emos is None and df_emos is None:
        return None

    blended: dict[str, float] = {}
    for k in _EMOTION_KEYS:
        vit_val = vit_emos.get(k, 0.0) if vit_emos else 0.0
        df_val = (df_emos.get(k, 0.0) / 100.0) if df_emos else 0.0

        if vit_emos and df_emos:
            blended[k] = _VIT_WEIGHT * vit_val + _DEEPFACE_WEIGHT * df_val
        elif vit_emos:
            blended[k] = vit_val
        else:
            blended[k] = df_val

    # Re-normalize so values sum to 1.0
    total = sum(blended.values())
    if total > 0:
        blended = {k: round(v / total, 4) for k, v in blended.items()}

    return blended


def _smooth_emotions(
    current: dict[str, float],
    previous: dict[str, float] | None,
    alpha: float = _SMOOTHING_ALPHA,
) -> dict[str, float]:
    """Exponential moving average over emotion scores to reduce flicker."""
    if previous is None:
        return current
    smoothed = {}
    for k in _EMOTION_KEYS:
        smoothed[k] = round(alpha * current.get(k, 0.0) + (1 - alpha) * previous.get(k, 0.0), 4)
    # Re-normalize
    total = sum(smoothed.values())
    if total > 0:
        smoothed = {k: round(v / total, 4) for k, v in smoothed.items()}
    return smoothed


def _engagement_score(emotions: dict[str, float]) -> float:
    """Compute engagement score (0.0-1.0) from normalized emotion probabilities.

    Emotions should be on 0-1 scale (already normalized by ensemble/smoothing).
    """
    raw = sum(emotions.get(k, 0.0) * w for k, w in _EMOTION_WEIGHTS.items())
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
    ):
        self.camera_index = camera_index
        self.sample_interval = sample_interval
        self.buffer_size = buffer_size
        self.baseline_frames = baseline_frames
        self.model_path = model_path

        self._vit_pipe = None
        self._smoothed_emotions: dict[str, float] | None = None
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

        # Initialize both emotion models
        self._vit_pipe = hf_pipeline("image-classification", model=_VIT_MODEL)
        # Warm up DeepFace by running a dummy analyze
        _dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        DeepFace.analyze(_dummy, actions=["emotion"], enforce_detection=False,
                         silent=True, detector_backend="skip")

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

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipe for face detection / presence
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect(mp_image)
                face_detected = len(results.face_landmarks) > 0
                presence = 1.0 if face_detected else 0.0

                # Ensemble emotion classification (ViT-FER + DeepFace)
                emotions: dict[str, float] | None = None
                dominant_emotion: str | None = None
                face_score: float | None = None

                if face_detected:
                    enhanced = _preprocess_frame(frame)

                    # ViT-FER pass
                    vit_emos: dict[str, float] | None = None
                    pil_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
                    vit_results = self._vit_pipe(pil_image, top_k=7)
                    if vit_results:
                        vit_emos = _vit_results_to_emotions(vit_results)

                    # DeepFace pass
                    df_emos: dict[str, float] | None = None
                    try:
                        df_results = DeepFace.analyze(
                            enhanced, actions=["emotion"],
                            enforce_detection=False, silent=True,
                            detector_backend="skip",
                        )
                        if df_results:
                            result = df_results[0] if isinstance(df_results, list) else df_results
                            df_emos = result["emotion"]
                    except Exception:
                        pass

                    # Blend and smooth
                    raw_ensemble = _ensemble_emotions(vit_emos, df_emos)
                    if raw_ensemble:
                        emotions = _smooth_emotions(raw_ensemble, self._smoothed_emotions)
                        self._smoothed_emotions = emotions
                        dominant_emotion = max(emotions, key=emotions.get)  # type: ignore[arg-type]
                        face_score = _engagement_score(emotions)

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
                    emotions=emotions,
                    dominant_emotion=dominant_emotion,
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
