from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .models import Baseline, HeadPose, ReactionFrame, SignalSource
from .scoring import COLLAPSED_KEYS, RAW_TO_COLLAPSED, capture_baseline, emotion_confidence


RAW_EMOTION_KEYS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
SMOOTHING_ALPHA = 0.35
DEFAULT_FACE_MODEL_PATH = Path(__file__).resolve().parents[2] / "face_landmarker.task"


def collapse_raw_emotions(raw_percentages: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    raw = {key: float(raw_percentages.get(key, 0.0)) / 100.0 for key in RAW_EMOTION_KEYS}
    raw_total = sum(raw.values())
    if raw_total > 0:
        raw = {key: round(value / raw_total, 4) for key, value in raw.items()}

    collapsed = {key: 0.0 for key in COLLAPSED_KEYS}
    for key, value in raw.items():
        collapsed[RAW_TO_COLLAPSED.get(key, "disinterested")] += value
    collapsed_total = sum(collapsed.values())
    if collapsed_total > 0:
        collapsed = {key: round(value / collapsed_total, 4) for key, value in collapsed.items()}
    return raw, collapsed


def smooth_emotions(
    current: dict[str, float],
    previous: dict[str, float] | None,
    *,
    alpha: float = SMOOTHING_ALPHA,
) -> dict[str, float]:
    if previous is None:
        return dict(current)
    smoothed = {
        key: round((alpha * current.get(key, 0.0)) + ((1.0 - alpha) * previous.get(key, 0.0)), 4)
        for key in COLLAPSED_KEYS
    }
    total = sum(smoothed.values())
    if total > 0:
        smoothed = {key: round(value / total, 4) for key, value in smoothed.items()}
    return smoothed


def engagement_score(emotions: dict[str, float]) -> float:
    return round(max(0.0, min(1.0, emotions.get("happy", 0.0))), 3)


class WebcamWorker:
    def __init__(
        self,
        *,
        camera_index: int = 0,
        sample_interval: float = 1.0,
        buffer_size: int = 120,
        baseline_frames: int = 3,
        model_path: str | os.PathLike[str] = DEFAULT_FACE_MODEL_PATH,
    ) -> None:
        self.camera_index = camera_index
        self.sample_interval = sample_interval
        self.buffer_size = buffer_size
        self.baseline_frames = baseline_frames
        self.model_path = str(model_path)
        self._frames: deque[ReactionFrame] = deque(maxlen=buffer_size)
        self._baseline: Baseline | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._smoothed_emotions: dict[str, float] | None = None

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
                "Download face_landmarker.task before enabling webcam reactions."
            )
        self._ensure_optional_dependencies()
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
        try:
            self._run_capture_loop()
        except Exception as exc:
            self._error = str(exc)
            self._running = False

    def _run_capture_loop(self) -> None:
        cv2, mp, np, deepface = self._optional_modules()
        base_options = mp.tasks.BaseOptions
        face_landmarker = mp.tasks.vision.FaceLandmarker
        options_type = mp.tasks.vision.FaceLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self._error = f"Could not open webcam index {self.camera_index}"
            self._running = False
            return

        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        deepface.analyze(dummy, actions=["emotion"], enforce_detection=False, silent=True, detector_backend="skip")
        previous_pose: HeadPose | None = None
        baseline_buffer: list[ReactionFrame] = []
        options = options_type(
            base_options=base_options(model_asset_path=self.model_path),
            running_mode=running_mode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with face_landmarker.create_from_options(options) as landmarker:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(self.sample_interval)
                    continue
                reaction_frame, previous_pose = self._frame_to_reaction(
                    frame,
                    previous_pose,
                    cv2=cv2,
                    mp=mp,
                    landmarker=landmarker,
                    deepface=deepface,
                    np=np,
                )
                with self._lock:
                    self._frames.append(reaction_frame)
                if self._baseline is None:
                    baseline_buffer.append(reaction_frame)
                    if len(baseline_buffer) >= self.baseline_frames:
                        self._baseline = capture_baseline(baseline_buffer)
                time.sleep(self.sample_interval)
        cap.release()

    def _frame_to_reaction(self, frame: Any, previous_pose: HeadPose | None, **deps: Any) -> tuple[ReactionFrame, HeadPose | None]:
        cv2 = deps["cv2"]
        mp = deps["mp"]
        landmarker = deps["landmarker"]
        deepface = deps["deepface"]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)
        if not results.face_landmarks:
            return ReactionFrame(timestamp=time.time(), presence=0.0, source=SignalSource.WEBCAM), None

        landmarks = results.face_landmarks[0]
        image_height, image_width = frame.shape[:2]
        head_pose = self._estimate_head_pose(landmarks, image_width, image_height, deps["cv2"], deps["np"])
        movement = self._head_movement(previous_pose, head_pose)
        face_crop = self._face_crop(frame, landmarks, image_width, image_height)
        raw_emotions: dict[str, float] | None = None
        collapsed_emotions: dict[str, float] | None = None
        dominant_emotion: str | None = None
        face_score: float | None = None
        confidence: float | None = None
        try:
            enhanced = self._preprocess_frame(face_crop, cv2)
            analyzed = deepface.analyze(
                enhanced,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
                detector_backend="skip",
            )
            if analyzed:
                result = analyzed[0] if isinstance(analyzed, list) else analyzed
                raw_emotions, collapsed = collapse_raw_emotions(result["emotion"])
                collapsed_emotions = smooth_emotions(collapsed, self._smoothed_emotions)
                self._smoothed_emotions = collapsed_emotions
                dominant_emotion = max(raw_emotions, key=raw_emotions.get)
                face_score = engagement_score(collapsed_emotions)
                confidence = emotion_confidence(collapsed_emotions)
        except Exception:
            pass
        return (
            ReactionFrame(
                timestamp=time.time(),
                presence=1.0,
                movement=movement,
                head_pose=head_pose,
                face=face_score,
                raw_emotions=raw_emotions,
                emotions=collapsed_emotions,
                dominant_emotion=dominant_emotion,
                emotion_confidence=confidence,
                source=SignalSource.WEBCAM,
            ),
            head_pose,
        )

    @staticmethod
    def _ensure_optional_dependencies() -> None:
        try:
            WebcamWorker._optional_modules()
        except ImportError as exc:
            raise RuntimeError(
                "Webcam reactions require optional dependencies. Install the backend with the 'reactions' extra."
            ) from exc

    @staticmethod
    def _optional_modules() -> tuple[Any, Any, Any, Any]:
        import cv2
        import mediapipe as mp
        import numpy as np
        from deepface import DeepFace

        return cv2, mp, np, DeepFace

    @staticmethod
    def _face_crop(frame: Any, landmarks: list[Any], image_width: int, image_height: int) -> Any:
        xs = [landmark.x * image_width for landmark in landmarks]
        ys = [landmark.y * image_height for landmark in landmarks]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        margin = int(0.2 * max(1, x2 - x1))
        crop = frame[max(0, y1 - margin) : min(image_height, y2 + margin), max(0, x1 - margin) : min(image_width, x2 + margin)]
        return crop if getattr(crop, "size", 0) else frame

    @staticmethod
    def _preprocess_frame(frame: Any, cv2: Any) -> Any:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        return cv2.cvtColor(cv2.merge([lightness, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _head_movement(previous_pose: HeadPose | None, current_pose: HeadPose | None) -> float:
        if previous_pose is None or current_pose is None:
            return 0.0
        delta = (
            (current_pose.yaw - previous_pose.yaw) ** 2
            + (current_pose.pitch - previous_pose.pitch) ** 2
            + (current_pose.roll - previous_pose.roll) ** 2
        ) ** 0.5
        return round(min(1.0, delta / 30.0), 3)

    @staticmethod
    def _estimate_head_pose(landmarks: list[Any], image_width: int, image_height: int, cv2: Any, np: Any) -> HeadPose | None:
        face_model = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float64,
        )
        landmark_indices = [1, 199, 33, 263, 61, 291]
        image_points = np.array(
            [(landmarks[index].x * image_width, landmarks[index].y * image_height) for index in landmark_indices],
            dtype=np.float64,
        )
        camera_matrix = np.array(
            [[float(image_width), 0, image_width / 2], [0, float(image_width), image_height / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        ok, rotation_vector, _ = cv2.solvePnP(
            face_model,
            image_points,
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        nose_2d, _ = cv2.projectPoints(
            np.array([[0.0, 0.0, 1000.0]], dtype=np.float64).reshape(1, 1, 3),
            rotation_vector,
            np.zeros((3, 1)),
            camera_matrix,
            np.zeros((4, 1)),
        )
        nose_tip = image_points[0]
        projected = (float(nose_2d[0, 0, 0]), float(nose_2d[0, 0, 1]))
        yaw = float((projected[0] - nose_tip[0]) / image_width * 90.0)
        pitch = float((projected[1] - nose_tip[1]) / image_height * 90.0)
        left_eye = image_points[2]
        right_eye = image_points[3]
        roll = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
        return HeadPose(
            yaw=round(float(np.clip(yaw, -90, 90)), 1),
            pitch=round(float(np.clip(pitch, -90, 90)), 1),
            roll=round(float(np.clip(roll, -90, 90)), 1),
        )
