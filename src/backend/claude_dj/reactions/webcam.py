from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .hud import draw_vibedj_hud
from .reaction import (
    COLLAPSED_KEYS,
    RAW_TO_COLLAPSED,
    HeadPose,
    LandmarkExpression,
    ReactionFrame,
    SignalSource,
    capture_baseline,
    emotion_confidence,
)


DEFAULT_FACE_MODEL_PATH = Path(__file__).resolve().parents[2] / "face_landmarker.task"
RAW_EMOTION_KEYS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
SMOOTHING_ALPHA = 0.35

FACE_3D_MODEL = [
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
]
LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]

LM_UPPER_LIP = 13
LM_LOWER_LIP = 14
LM_LEFT_MOUTH = 61
LM_RIGHT_MOUTH = 291
LM_NOSE_TIP = 1
LM_LEFT_EYE = {
    "outer": 33,
    "inner": 133,
    "upper1": 159,
    "lower1": 145,
    "upper2": 160,
    "lower2": 144,
}
LM_RIGHT_EYE = {
    "outer": 263,
    "inner": 362,
    "upper1": 386,
    "lower1": 374,
    "upper2": 387,
    "lower2": 373,
}
LM_LEFT_BROW = [70, 63, 105]
LM_RIGHT_BROW = [300, 293, 334]
LM_LEFT_EYE_TOP = 159
LM_RIGHT_EYE_TOP = 386


def _dist(landmark_a: Any, landmark_b: Any, width: int, height: int) -> float:
    dx = (landmark_a.x - landmark_b.x) * width
    dy = (landmark_a.y - landmark_b.y) * height
    return float((dx**2 + dy**2) ** 0.5)


def _estimate_head_pose(landmarks: list[Any], image_width: int, image_height: int) -> HeadPose | None:
    cv2, _, np, _ = WebcamWorker._optional_modules()
    image_points = np.array(
        [(landmarks[index].x * image_width, landmarks[index].y * image_height) for index in LANDMARK_INDICES],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [[float(image_width), 0, image_width / 2], [0, float(image_width), image_height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    ok, rotation_vector, _ = cv2.solvePnP(
        np.array(FACE_3D_MODEL, dtype=np.float64),
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
    roll = float(
        np.degrees(
            np.arctan2(
                image_points[3][1] - image_points[2][1],
                image_points[3][0] - image_points[2][0],
            )
        )
    )
    return HeadPose(
        yaw=round(float(np.clip(yaw, -90, 90)), 1),
        pitch=round(float(np.clip(pitch, -90, 90)), 1),
        roll=round(float(np.clip(roll, -90, 90)), 1),
    )


def _head_movement(previous_pose: HeadPose | None, current_pose: HeadPose | None) -> float:
    if previous_pose is None or current_pose is None:
        return 0.0
    delta = (
        (current_pose.yaw - previous_pose.yaw) ** 2
        + (current_pose.pitch - previous_pose.pitch) ** 2
        + (current_pose.roll - previous_pose.roll) ** 2
    ) ** 0.5
    return round(min(1.0, delta / 30.0), 3)


def _compute_smile(landmarks: list[Any], image_width: int, image_height: int) -> float:
    left_corner = landmarks[LM_LEFT_MOUTH]
    right_corner = landmarks[LM_RIGHT_MOUTH]
    lip_center = landmarks[LM_UPPER_LIP]
    avg_rise = (
        ((lip_center.y - left_corner.y) * image_height)
        + ((lip_center.y - right_corner.y) * image_height)
    ) / 2.0
    mouth_width = _dist(left_corner, right_corner, image_width, image_height)
    if mouth_width < 1.0:
        return 0.0
    face_ref = _dist(landmarks[LANDMARK_INDICES[2]], landmarks[LANDMARK_INDICES[3]], image_width, image_height)
    if face_ref < 1.0:
        face_ref = mouth_width
    width_ratio = mouth_width / face_ref
    rise_ratio = avg_rise / mouth_width
    raw = rise_ratio * 0.7 + (width_ratio - 0.5) * 0.3
    return round(max(0.0, min(1.0, raw * 5.0)), 3)


def _compute_mouth_ratio(landmarks: list[Any], image_width: int, image_height: int) -> float:
    mouth_height = _dist(landmarks[LM_UPPER_LIP], landmarks[LM_LOWER_LIP], image_width, image_height)
    mouth_width = _dist(landmarks[LM_LEFT_MOUTH], landmarks[LM_RIGHT_MOUTH], image_width, image_height)
    if mouth_width < 1.0:
        return 0.0
    return round(max(0.0, min(1.0, mouth_height / mouth_width)), 3)


def _compute_ear(landmarks: list[Any], image_width: int, image_height: int) -> float:
    def eye_ear(indices: dict[str, int]) -> float:
        vertical_1 = _dist(landmarks[indices["upper1"]], landmarks[indices["lower1"]], image_width, image_height)
        vertical_2 = _dist(landmarks[indices["upper2"]], landmarks[indices["lower2"]], image_width, image_height)
        horizontal = _dist(landmarks[indices["outer"]], landmarks[indices["inner"]], image_width, image_height)
        if horizontal < 1.0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    return round(max(0.0, min(1.0, (eye_ear(LM_LEFT_EYE) + eye_ear(LM_RIGHT_EYE)) / 2.0)), 3)


def _compute_brow_height(landmarks: list[Any], image_width: int, image_height: int) -> float:
    inter_eye = _dist(landmarks[33], landmarks[263], image_width, image_height)
    if inter_eye < 1.0:
        return 0.5
    left_brow_y = float(sum(landmarks[index].y for index in LM_LEFT_BROW)) / len(LM_LEFT_BROW)
    right_brow_y = float(sum(landmarks[index].y for index in LM_RIGHT_BROW)) / len(LM_RIGHT_BROW)
    left_gap = (float(landmarks[LM_LEFT_EYE_TOP].y) - left_brow_y) * image_height
    right_gap = (float(landmarks[LM_RIGHT_EYE_TOP].y) - right_brow_y) * image_height
    ratio = ((left_gap + right_gap) / 2.0) / inter_eye
    return round(max(0.0, min(1.0, (ratio - 0.10) / 0.20)), 3)


def _compute_landmark_expression(landmarks: list[Any], image_width: int, image_height: int) -> LandmarkExpression:
    return LandmarkExpression(
        smile=_compute_smile(landmarks, image_width, image_height),
        mouth_open=_compute_mouth_ratio(landmarks, image_width, image_height),
        ear=_compute_ear(landmarks, image_width, image_height),
        brow_height=_compute_brow_height(landmarks, image_width, image_height),
    )


def _preprocess_frame(frame: Any) -> Any:
    cv2, _, _, _ = WebcamWorker._optional_modules()
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge([lightness, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def _deepface_to_emotions(df_emotions: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    raw = {key: float(df_emotions.get(key, 0.0)) / 100.0 for key in RAW_EMOTION_KEYS}
    raw_total = sum(raw.values())
    if raw_total > 0:
        raw = {key: round(value / raw_total, 4) for key, value in raw.items()}
    collapsed = {key: 0.0 for key in COLLAPSED_KEYS}
    for raw_key, score in raw.items():
        collapsed[RAW_TO_COLLAPSED.get(raw_key, "disinterested")] += score
    collapsed_total = sum(collapsed.values())
    if collapsed_total > 0:
        collapsed = {key: round(value / collapsed_total, 4) for key, value in collapsed.items()}
    return raw, collapsed


def _smooth_emotions(
    current: dict[str, float],
    previous: dict[str, float] | None,
    alpha: float = SMOOTHING_ALPHA,
    confidence: float = 0.5,
) -> dict[str, float]:
    if previous is None:
        return dict(current)
    alpha_effective = alpha + (1.0 - alpha) * confidence
    smoothed = {
        key: round(
            (alpha_effective * current.get(key, 0.0))
            + ((1.0 - alpha_effective) * previous.get(key, 0.0)),
            4,
        )
        for key in COLLAPSED_KEYS
    }
    total = sum(smoothed.values())
    if total > 0:
        smoothed = {key: round(value / total, 4) for key, value in smoothed.items()}
    return smoothed


def _engagement_score(
    emotions: dict[str, float],
    landmark_expression: LandmarkExpression | None = None,
    movement: float | None = None,
    head_pose: HeadPose | None = None,
) -> float:
    score = 0.5
    if landmark_expression is not None:
        score += landmark_expression.smile * 0.35
        if landmark_expression.mouth_open > 0.08:
            score += landmark_expression.mouth_open * 0.1
        brow_delta = landmark_expression.brow_height - 0.5
        if brow_delta < -0.15:
            score -= 0.15
        elif brow_delta > 0.1:
            score += 0.05
    if movement is not None and movement > 0.03:
        score += min(movement * 0.25, 0.15)
    if head_pose is not None and abs(head_pose.yaw) > 20:
        score -= 0.2
    score += emotions.get("happy", 0.0) * 0.15
    return round(max(0.0, min(1.0, score)), 3)


class WebcamWorker:
    def __init__(
        self,
        *,
        camera_index: int = 0,
        sample_interval: float = 1.0,
        buffer_size: int = 120,
        baseline_frames: int = 3,
        model_path: str | os.PathLike[str] = DEFAULT_FACE_MODEL_PATH,
        show_preview: bool = False,
        preview_window_name: str = "VibeDJ",
        preview_fps: float = 30.0,
    ) -> None:
        self.camera_index = camera_index
        self.sample_interval = sample_interval
        self.preview_fps = preview_fps
        self.buffer_size = buffer_size
        self.baseline_frames = baseline_frames
        self.model_path = str(model_path)
        self.show_preview = show_preview
        self.preview_window_name = preview_window_name
        self._smoothed_emotions: dict[str, float] | None = None
        self._frames: deque[ReactionFrame] = deque(maxlen=buffer_size)
        self._baseline = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._preview_frame: Any | None = None
        self._preview_reaction_frame: ReactionFrame | None = None
        self._pitch_history: deque[float] = deque(maxlen=150)

    @property
    def baseline(self):
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
        self._preflight_camera_access()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.close_preview_window()

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
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self._error = f"Could not open webcam index {self.camera_index}"
            self._running = False
            return

        try:
            dummy = np.zeros((48, 48, 3), dtype=np.uint8)
            deepface.analyze(
                dummy,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
                detector_backend="skip",
            )
        except Exception as exc:
            self._error = str(exc)
            self._running = False
            cap.release()
            return

        previous_pose: HeadPose | None = None
        baseline_buffer: list[ReactionFrame] = []
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            next_analysis_at = 0.0
            preview_interval = 1.0 / max(1.0, self.preview_fps)
            while self._running:
                loop_started = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(preview_interval)
                    continue
                reaction_frame = None
                if loop_started >= next_analysis_at:
                    reaction_frame, previous_pose = self._frame_to_reaction(
                        frame,
                        landmarker,
                        previous_pose,
                        cv2,
                        mp,
                        deepface,
                    )
                    with self._lock:
                        self._frames.append(reaction_frame)
                    if self._baseline is None:
                        baseline_buffer.append(reaction_frame)
                        if len(baseline_buffer) >= self.baseline_frames:
                            self._baseline = capture_baseline(baseline_buffer)
                    next_analysis_at = loop_started + max(0.0, self.sample_interval)
                if self.show_preview:
                    self._store_preview_frame(frame, reaction_frame)
                elapsed = time.monotonic() - loop_started
                time.sleep(max(0.0, preview_interval - elapsed))
        cap.release()

    def _frame_to_reaction(
        self,
        frame: Any,
        landmarker: Any,
        previous_pose: HeadPose | None,
        cv2: Any,
        mp: Any,
        deepface: Any,
    ) -> tuple[ReactionFrame, HeadPose | None]:
        image_height, image_width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)
        if not results.face_landmarks:
            reaction_frame = ReactionFrame(timestamp=time.time(), presence=0.0, source=SignalSource.WEBCAM)
            return reaction_frame, None

        landmarks = results.face_landmarks[0]
        head_pose = _estimate_head_pose(landmarks, image_width, image_height)
        movement = _head_movement(previous_pose, head_pose)
        landmark_expression = _compute_landmark_expression(landmarks, image_width, image_height)
        xs = [landmark.x * image_width for landmark in landmarks]
        ys = [landmark.y * image_height for landmark in landmarks]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        frame_area = image_width * image_height
        face_area = float((x2 - x1) * (y2 - y1)) / frame_area if frame_area > 0 else 0.0
        margin = int(0.2 * max(1, x2 - x1))
        face_crop = frame[
            max(0, y1 - margin) : min(image_height, y2 + margin),
            max(0, x1 - margin) : min(image_width, x2 + margin),
        ]
        if getattr(face_crop, "size", 0) == 0:
            face_crop = frame

        raw_emotions = None
        collapsed_emotions = None
        dominant_emotion = None
        face_score = None
        face_confidence = None
        try:
            result = deepface.analyze(
                _preprocess_frame(face_crop),
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
                detector_backend="skip",
            )
            if result:
                result_data = result[0] if isinstance(result, list) else result
                raw_emotions, collapsed = _deepface_to_emotions(result_data["emotion"])
                face_confidence = emotion_confidence(collapsed)
                collapsed_emotions = _smooth_emotions(collapsed, self._smoothed_emotions, confidence=face_confidence)
                self._smoothed_emotions = collapsed_emotions
                face_score = _engagement_score(
                    collapsed_emotions,
                    landmark_expression=landmark_expression,
                    movement=movement,
                    head_pose=head_pose,
                )
                dominant_emotion = max(raw_emotions, key=raw_emotions.get)
        except Exception:
            pass

        reaction_frame = ReactionFrame(
            timestamp=time.time(),
            presence=1.0,
            movement=movement,
            head_pose=head_pose,
            face=face_score,
            raw_emotions=raw_emotions,
            emotions=collapsed_emotions,
            dominant_emotion=dominant_emotion,
            emotion_confidence=face_confidence,
            landmark_expression=landmark_expression,
            face_area=face_area,
            source=SignalSource.WEBCAM,
        )
        return reaction_frame, head_pose

    def _store_preview_frame(self, frame: Any, reaction_frame: ReactionFrame | None) -> None:
        with self._lock:
            self._preview_frame = frame.copy()
            if reaction_frame is not None:
                self._preview_reaction_frame = reaction_frame
            if reaction_frame is not None and reaction_frame.head_pose is not None:
                self._pitch_history.append(float(reaction_frame.head_pose.pitch))

    def pump_preview_window(self) -> bool:
        if not self.show_preview:
            return False
        with self._lock:
            frame = self._preview_frame.copy() if self._preview_frame is not None else None
            reaction_frame = self._preview_reaction_frame
            pitch_history = list(self._pitch_history)
        if frame is None:
            return False
        import cv2

        frame = draw_vibedj_hud(frame, reaction_frame, pitch_history)
        cv2.imshow(self.preview_window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._running = False
            self.close_preview_window()
            return True
        return False

    def close_preview_window(self) -> None:
        try:
            import cv2

            cv2.destroyWindow(self.preview_window_name)
        except Exception:
            pass

    @staticmethod
    def _ensure_optional_dependencies() -> None:
        try:
            WebcamWorker._optional_modules()
        except ImportError as exc:
            raise RuntimeError(
                "Webcam reactions require optional dependencies. Install the backend with the 'reactions' extra."
            ) from exc

    def _preflight_camera_access(self) -> None:
        cv2, _, _, _ = self._optional_modules()
        cap = cv2.VideoCapture(self.camera_index)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Could not open webcam index {self.camera_index}")
        finally:
            cap.release()

    @staticmethod
    def _optional_modules() -> tuple[Any, Any, Any, Any]:
        import cv2
        import mediapipe as mp
        import numpy as np
        from deepface import DeepFace

        return cv2, mp, np, DeepFace
