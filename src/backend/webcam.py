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

from reaction import (
    Baseline, COLLAPSED_KEYS, EMOTION_WEIGHTS, HeadPose, RAW_TO_COLLAPSED,
    ReactionFrame, SignalSource, capture_baseline, emotion_confidence,
)

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model path — sits alongside this file
_MODEL_PATH = str(Path(__file__).parent / "face_landmarker.task")

# ViT-FER model from HuggingFace (92.2% accuracy on FER2013+AffectNet)
_VIT_MODEL = "HardlyHumans/Facial-expression-detection"

# Raw 7-class emotion keys from the models (before collapsing to 2-state)
_RAW_EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Ensemble blend: ViT gets more weight due to higher accuracy.
_VIT_WEIGHT = 0.65
_DEEPFACE_WEIGHT = 0.35

# Temporal smoothing factor (exponential moving average).
# Lower = smoother but slower to react. 0.35 gives ~3-frame lag.
_SMOOTHING_ALPHA = 0.35


# 3D model points for a canonical face (mm), used with solvePnP.
# Indices: nose tip (1), chin (199), left eye outer (33),
# right eye outer (263), left mouth (61), right mouth (291).
_FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye outer corner
    (225.0, 170.0, -135.0),    # Right eye outer corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0),   # Right mouth corner
], dtype=np.float64)

_LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]


def _estimate_head_pose(
    landmarks: list, img_w: int, img_h: int,
) -> HeadPose | None:
    """Estimate head yaw/pitch/roll from MediaPipe face landmarks via solvePnP."""
    image_points = np.array([
        (landmarks[i].x * img_w, landmarks[i].y * img_h)
        for i in _LANDMARK_INDICES
    ], dtype=np.float64)

    focal_length = img_w
    camera_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(
        _FACE_3D_MODEL, image_points, camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    # Project nose tip to get pose angles directly from the rotation.
    # This avoids Euler angle convention issues with decomposition.
    nose_3d = np.array([0.0, 0.0, 1000.0])  # point along nose axis
    nose_2d, _ = cv2.projectPoints(
        nose_3d.reshape(1, 1, 3), rvec, np.zeros((3, 1)),
        camera_matrix, np.zeros((4, 1)),
    )
    p1 = image_points[0]  # nose tip in image
    p2 = (float(nose_2d[0, 0, 0]), float(nose_2d[0, 0, 1]))

    yaw = float((p2[0] - p1[0]) / img_w * 90)   # horizontal deflection → yaw
    pitch = float((p2[1] - p1[1]) / img_h * 90)  # vertical deflection → pitch

    # Roll from the eye line
    le = image_points[2]  # left eye
    re = image_points[3]  # right eye
    roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))

    return HeadPose(
        yaw=round(np.clip(yaw, -90, 90), 1),
        pitch=round(np.clip(pitch, -90, 90), 1),
        roll=round(np.clip(roll, -90, 90), 1),
    )


def _head_movement(prev_pose: HeadPose | None, curr_pose: HeadPose | None) -> float:
    """Head movement magnitude from pose delta (0.0–1.0).

    Uses the Euclidean distance of (yaw, pitch, roll) changes,
    normalized so ~30° total change maps to 1.0.
    """
    if prev_pose is None or curr_pose is None:
        return 0.0
    dy = curr_pose.yaw - prev_pose.yaw
    dp = curr_pose.pitch - prev_pose.pitch
    dr = curr_pose.roll - prev_pose.roll
    magnitude = (dy ** 2 + dp ** 2 + dr ** 2) ** 0.5
    return round(min(1.0, magnitude / 30.0), 3)


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE to normalize lighting before emotion classification."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


_VIT_LABEL_MAP: dict[str, str] = {
    "anger": "angry",
    "contempt": "disgust",  # closest canonical emotion
}


def _vit_results_to_emotions(vit_results: list[dict]) -> dict[str, float]:
    """Convert HuggingFace pipeline output to raw 7-class emotion dict (0-1).

    Handles label differences between ViT-FER (AffectNet labels) and our
    canonical FER2013 keys: "anger" → "angry", "contempt" → "disgust".
    """
    emos = {k: 0.0 for k in _RAW_EMOTION_KEYS}
    for item in vit_results:
        label = _VIT_LABEL_MAP.get(item["label"].lower(), item["label"].lower())
        if label in emos:
            emos[label] += item["score"]
    return emos


def _ensemble_emotions(
    vit_emos: dict[str, float] | None,
    df_emos: dict[str, float] | None,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Blend ViT-FER and DeepFace into raw 7-class and collapsed 3-state distributions.

    Returns (raw_7class, collapsed_3state). Both normalize to ~1.0.
    The raw distribution preserves the full emotional signal so the agent
    can distinguish angry from sad from surprised.
    """
    if vit_emos is None and df_emos is None:
        return None, None

    # Blend raw 7-class scores
    raw_blended: dict[str, float] = {}
    for k in _RAW_EMOTION_KEYS:
        vit_val = vit_emos.get(k, 0.0) if vit_emos else 0.0
        df_val = (df_emos.get(k, 0.0) / 100.0) if df_emos else 0.0

        if vit_emos and df_emos:
            raw_blended[k] = _VIT_WEIGHT * vit_val + _DEEPFACE_WEIGHT * df_val
        elif vit_emos:
            raw_blended[k] = vit_val
        else:
            raw_blended[k] = df_val

    # Normalize raw 7-class to sum to 1.0
    raw_total = sum(raw_blended.values())
    if raw_total > 0:
        raw_blended = {k: round(v / raw_total, 4) for k, v in raw_blended.items()}

    # Collapse into 3-state: happy / neutral / disinterested
    collapsed: dict[str, float] = {k: 0.0 for k in COLLAPSED_KEYS}
    for raw_key, score in raw_blended.items():
        target = RAW_TO_COLLAPSED.get(raw_key, "disinterested")
        collapsed[target] += score

    # Normalize collapsed
    col_total = sum(collapsed.values())
    if col_total > 0:
        collapsed = {k: round(v / col_total, 4) for k, v in collapsed.items()}

    return raw_blended, collapsed


def _smooth_emotions(
    current: dict[str, float],
    previous: dict[str, float] | None,
    alpha: float = _SMOOTHING_ALPHA,
) -> dict[str, float]:
    """Exponential moving average over 2-state emotion scores to reduce flicker."""
    if previous is None:
        return current
    smoothed = {}
    for k in COLLAPSED_KEYS:
        smoothed[k] = round(alpha * current.get(k, 0.0) + (1 - alpha) * previous.get(k, 0.0), 4)
    # Re-normalize
    total = sum(smoothed.values())
    if total > 0:
        smoothed = {k: round(v / total, 4) for k, v in smoothed.items()}
    return smoothed


def _engagement_score(emotions: dict[str, float]) -> float:
    """Compute engagement score (0.0-1.0) from 2-state emotion probabilities.

    With only happy/disinterested, this is essentially the happy probability
    mapped to 0-1 engagement.
    """
    happy = emotions.get("happy", 0.0)
    return round(max(0.0, min(1.0, happy)), 3)


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
        self._error: str | None = None

    @property
    def baseline(self) -> Baseline | None:
        return self._baseline

    @property
    def error(self) -> str | None:
        return self._error

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
        try:
            self._vit_pipe = hf_pipeline("image-classification", model=_VIT_MODEL)
            # Warm up DeepFace by running a dummy analyze
            _dummy = np.zeros((48, 48, 3), dtype=np.uint8)
            DeepFace.analyze(_dummy, actions=["emotion"], enforce_detection=False,
                             silent=True, detector_backend="skip")
        except Exception as e:
            self._error = str(e)
            self._running = False
            cap.release()
            return

        prev_pose: HeadPose | None = None
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

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipe for face detection / presence
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect(mp_image)
                face_detected = len(results.face_landmarks) > 0
                presence = 1.0 if face_detected else 0.0

                # Head pose and movement from landmarks
                head_pose: HeadPose | None = None
                movement: float | None = None
                raw_emotions: dict[str, float] | None = None
                collapsed_emotions: dict[str, float] | None = None
                dominant_emotion: str | None = None
                face_score: float | None = None
                face_conf: float | None = None

                if face_detected:
                    lmarks = results.face_landmarks[0]
                    ih, iw = frame.shape[:2]

                    # Head pose via solvePnP on 6 key landmarks
                    head_pose = _estimate_head_pose(lmarks, iw, ih)
                    movement = _head_movement(prev_pose, head_pose)
                    prev_pose = head_pose

                    # Crop face from landmarks for emotion accuracy
                    xs = [lm.x * iw for lm in lmarks]
                    ys = [lm.y * ih for lm in lmarks]
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))
                    margin = int(0.2 * (x2 - x1))
                    face_crop = frame[
                        max(0, y1 - margin):min(ih, y2 + margin),
                        max(0, x1 - margin):min(iw, x2 + margin),
                    ]
                    if face_crop.size == 0:
                        face_crop = frame

                    enhanced = _preprocess_frame(face_crop)

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

                    # Blend → raw 7-class + collapsed 3-state
                    raw_ensemble, collapsed_ensemble = _ensemble_emotions(vit_emos, df_emos)
                    if collapsed_ensemble:
                        collapsed_emotions = _smooth_emotions(
                            collapsed_ensemble, self._smoothed_emotions,
                        )
                        self._smoothed_emotions = collapsed_emotions
                        face_score = _engagement_score(collapsed_emotions)
                        face_conf = emotion_confidence(collapsed_emotions)
                    raw_emotions = raw_ensemble
                    if raw_emotions:
                        dominant_emotion = max(raw_emotions, key=raw_emotions.get)  # type: ignore[arg-type]
                else:
                    prev_pose = None  # reset pose tracking when face lost

                reaction_frame = ReactionFrame(
                    timestamp=time.time(),
                    presence=presence,
                    movement=movement,
                    head_pose=head_pose,
                    face=face_score,
                    raw_emotions=raw_emotions,
                    emotions=collapsed_emotions,
                    dominant_emotion=dominant_emotion,
                    emotion_confidence=face_conf,
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
