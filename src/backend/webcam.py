"""Webcam reaction worker for ClaudeDJ.

Captures frames from the webcam at ~1fps, extracts presence, movement,
and facial expression signals, and produces ReactionFrames. Runs in a
background thread so it never blocks playback (P6).

Uses MediaPipe FaceLandmarker for face detection/presence and DeepFace
for emotion classification, with temporal smoothing to reduce
frame-to-frame noise.

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

from reaction import (
    Baseline, COLLAPSED_KEYS, EMOTION_WEIGHTS, HeadPose, LandmarkExpression,
    RAW_TO_COLLAPSED, ReactionFrame, SignalSource, capture_baseline, emotion_confidence,
)

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model path — sits alongside this file
_MODEL_PATH = str(Path(__file__).parent / "face_landmarker.task")

# Raw 7-class emotion keys from DeepFace (before collapsing to 3-state)
_RAW_EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

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


# ---------------------------------------------------------------------------
# Landmark expression features
# ---------------------------------------------------------------------------
# Landmark indices for expression geometry.
_LM_UPPER_LIP = 13
_LM_LOWER_LIP = 14
_LM_LEFT_MOUTH = 61
_LM_RIGHT_MOUTH = 291
_LM_NOSE_TIP = 1

# Eye landmarks (left/right).
_LM_LEFT_EYE = {"outer": 33, "inner": 133, "upper1": 159, "lower1": 145, "upper2": 160, "lower2": 144}
_LM_RIGHT_EYE = {"outer": 263, "inner": 362, "upper1": 386, "lower1": 374, "upper2": 387, "lower2": 373}

# Brow landmarks — inner/mid/outer for each brow, measured relative to eye top.
_LM_LEFT_BROW = [70, 63, 105]   # inner, mid, outer
_LM_RIGHT_BROW = [300, 293, 334]
_LM_LEFT_EYE_TOP = 159
_LM_RIGHT_EYE_TOP = 386


def _dist(lm_a, lm_b, w: int, h: int) -> float:
    """Pixel distance between two landmarks."""
    dx = (lm_a.x - lm_b.x) * w
    dy = (lm_a.y - lm_b.y) * h
    return float((dx ** 2 + dy ** 2) ** 0.5)


def _compute_smile(landmarks: list, img_w: int, img_h: int) -> float:
    """Smile score from lip corner geometry (0.0–1.0).

    Measures how much lip corners rise relative to lip center,
    normalized by mouth width. A neutral face gives ~0, a smile gives >0.5.
    """
    left_corner = landmarks[_LM_LEFT_MOUTH]
    right_corner = landmarks[_LM_RIGHT_MOUTH]
    lip_center = landmarks[_LM_UPPER_LIP]

    # Lip corners rising = corner y < lip center y (in image coords, y increases downward)
    left_rise = float(lip_center.y - left_corner.y) * img_h
    right_rise = float(lip_center.y - right_corner.y) * img_h
    avg_rise = (left_rise + right_rise) / 2

    # Normalize by mouth width for face-size invariance
    mouth_width = _dist(left_corner, right_corner, img_w, img_h)
    if mouth_width < 1.0:
        return 0.0

    # Also factor in mouth widening (smiles widen the mouth)
    nose = landmarks[_LM_NOSE_TIP]
    face_ref = _dist(landmarks[_LANDMARK_INDICES[2]], landmarks[_LANDMARK_INDICES[3]], img_w, img_h)
    if face_ref < 1.0:
        face_ref = mouth_width

    width_ratio = mouth_width / face_ref
    rise_ratio = avg_rise / mouth_width

    # Combine: rise is primary signal, width is secondary
    raw = rise_ratio * 0.7 + (width_ratio - 0.5) * 0.3
    # Scale so typical range maps to 0-1
    score = max(0.0, min(1.0, raw * 5.0))
    return round(score, 3)


def _compute_mouth_ratio(landmarks: list, img_w: int, img_h: int) -> float:
    """Mouth opening ratio (0.0–1.0).

    Ratio of vertical lip separation to mouth width.
    Detects singing along, jaw drop, or mouthing words.
    """
    upper = landmarks[_LM_UPPER_LIP]
    lower = landmarks[_LM_LOWER_LIP]
    left = landmarks[_LM_LEFT_MOUTH]
    right = landmarks[_LM_RIGHT_MOUTH]

    mouth_height = _dist(upper, lower, img_w, img_h)
    mouth_width = _dist(left, right, img_w, img_h)

    if mouth_width < 1.0:
        return 0.0

    ratio = mouth_height / mouth_width
    # Clamp to 0-1; typical range is 0.0 (closed) to ~0.8 (wide open)
    return round(max(0.0, min(1.0, ratio)), 3)


def _compute_ear(landmarks: list, img_w: int, img_h: int) -> float:
    """Eye Aspect Ratio averaged over both eyes (0.0–1.0).

    EAR = (v1 + v2) / (2 * h) per eye, averaged.
    Low EAR = eyes closing/squinting, high EAR = eyes wide open.
    """
    def _eye_ear(eye_idx: dict) -> float:
        v1 = _dist(landmarks[eye_idx["upper1"]], landmarks[eye_idx["lower1"]], img_w, img_h)
        v2 = _dist(landmarks[eye_idx["upper2"]], landmarks[eye_idx["lower2"]], img_w, img_h)
        h = _dist(landmarks[eye_idx["outer"]], landmarks[eye_idx["inner"]], img_w, img_h)
        if h < 1.0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    left_ear = _eye_ear(_LM_LEFT_EYE)
    right_ear = _eye_ear(_LM_RIGHT_EYE)
    ear = (left_ear + right_ear) / 2.0
    return round(max(0.0, min(1.0, ear)), 3)


def _compute_brow_height(landmarks: list, img_w: int, img_h: int) -> float:
    """Brow height relative to eyes (0.0=lowered/furrowed, 0.5=neutral, 1.0=raised).

    Measures average brow-to-eye-top distance, normalized by inter-eye distance.
    Raised brows = interest/surprise, lowered = focus/concentration/negative.
    """
    inter_eye = _dist(landmarks[33], landmarks[263], img_w, img_h)
    if inter_eye < 1.0:
        return 0.5

    # Average brow-to-eye distance for each side
    left_brow_y = float(sum(landmarks[i].y for i in _LM_LEFT_BROW)) / len(_LM_LEFT_BROW)
    right_brow_y = float(sum(landmarks[i].y for i in _LM_RIGHT_BROW)) / len(_LM_RIGHT_BROW)
    left_eye_y = float(landmarks[_LM_LEFT_EYE_TOP].y)
    right_eye_y = float(landmarks[_LM_RIGHT_EYE_TOP].y)

    # Distance in pixels (brow is above eye, so eye_y > brow_y in image coords)
    left_gap = (left_eye_y - left_brow_y) * img_h
    right_gap = (right_eye_y - right_brow_y) * img_h
    avg_gap = (left_gap + right_gap) / 2.0

    # Normalize by inter-eye distance — typical ratio is ~0.15-0.25
    ratio = avg_gap / inter_eye
    # Map to 0-1 where 0.18 is neutral
    score = (ratio - 0.10) / 0.20  # 0.10 = fully lowered, 0.30 = fully raised
    return round(max(0.0, min(1.0, score)), 3)


def _compute_landmark_expression(
    landmarks: list, img_w: int, img_h: int,
) -> LandmarkExpression:
    """Compute all landmark expression features in one pass."""
    return LandmarkExpression(
        smile=_compute_smile(landmarks, img_w, img_h),
        mouth_open=_compute_mouth_ratio(landmarks, img_w, img_h),
        ear=_compute_ear(landmarks, img_w, img_h),
        brow_height=_compute_brow_height(landmarks, img_w, img_h),
    )


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE to normalize lighting before emotion classification."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _deepface_to_emotions(
    df_emos: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Convert DeepFace emotion percentages to raw 7-class and collapsed 3-state.

    Returns (raw_7class, collapsed_3state). Both normalize to ~1.0.
    DeepFace returns percentages (0-100), so we divide by 100.
    """
    # Normalize raw 7-class to sum to 1.0
    raw: dict[str, float] = {}
    for k in _RAW_EMOTION_KEYS:
        raw[k] = df_emos.get(k, 0.0) / 100.0

    raw_total = sum(raw.values())
    if raw_total > 0:
        raw = {k: round(v / raw_total, 4) for k, v in raw.items()}

    # Collapse into 3-state: happy / neutral / disinterested
    collapsed: dict[str, float] = {k: 0.0 for k in COLLAPSED_KEYS}
    for raw_key, score in raw.items():
        target = RAW_TO_COLLAPSED.get(raw_key, "disinterested")
        collapsed[target] += score

    # Normalize collapsed
    col_total = sum(collapsed.values())
    if col_total > 0:
        collapsed = {k: round(v / col_total, 4) for k, v in collapsed.items()}

    return raw, collapsed


def _smooth_emotions(
    current: dict[str, float],
    previous: dict[str, float] | None,
    alpha: float = _SMOOTHING_ALPHA,
    confidence: float = 0.5,
) -> dict[str, float]:
    """Adaptive EMA over emotion scores — responds faster when confidence is high.

    α_effective = α_base + (1 - α_base) × confidence
    High confidence (peaked, 0.8+): α ≈ 0.87 → responds in ~1 frame
    Low confidence (flat, 0.2):     α ≈ 0.48 → smooths over ~2-3 frames
    Very low confidence:            heavy smoothing, basically ignores the reading
    """
    if previous is None:
        return current
    alpha_effective = alpha + (1.0 - alpha) * confidence
    smoothed = {}
    for k in COLLAPSED_KEYS:
        smoothed[k] = round(
            alpha_effective * current.get(k, 0.0)
            + (1 - alpha_effective) * previous.get(k, 0.0),
            4,
        )
    # Re-normalize
    total = sum(smoothed.values())
    if total > 0:
        smoothed = {k: round(v / total, 4) for k, v in smoothed.items()}
    return smoothed


def _engagement_score(
    emotions: dict[str, float],
    landmark_expr: LandmarkExpression | None = None,
    movement: float | None = None,
    head_pose: HeadPose | None = None,
) -> float:
    """Compute engagement score (0.0-1.0) from landmarks, head pose, and emotions.

    Primary signals (landmarks):
    - Smile → happy (strongest positive signal)
    - Mouth open → singing along → happy
    - Brow lowered → disinterested, brow neutral/raised → neutral/interested

    Secondary signals (head):
    - Nodding (movement) → engaged → pushes toward happy
    - Looking away (high yaw) → disinterested

    Tertiary signal (DeepFace):
    - Happy boosts score, but given low weight since it's unreliable for subtle expressions
    """
    # Start at neutral
    score = 0.5

    # Landmark smile is the strongest "likes it" signal
    if landmark_expr is not None:
        # Smile: even micro-smiles (>0.05) push positive
        score += landmark_expr.smile * 0.35

        # Mouth open above resting = singing along / reacting
        if landmark_expr.mouth_open > 0.08:
            score += landmark_expr.mouth_open * 0.1

        # Brow position: lowered = disengaged, raised = interested
        brow_delta = landmark_expr.brow_height - 0.5
        if brow_delta < -0.15:
            # Lowered/furrowed brows → disinterested
            score -= 0.15
        elif brow_delta > 0.1:
            # Raised brows → interested/surprised
            score += 0.05

    # Head nodding = engagement
    if movement is not None and movement > 0.03:
        score += min(movement * 0.25, 0.15)

    # Looking away = disinterested
    if head_pose is not None and abs(head_pose.yaw) > 20:
        score -= 0.2

    # DeepFace as weak supporting signal
    happy = emotions.get("happy", 0.0)
    score += happy * 0.15

    return round(max(0.0, min(1.0, score)), 3)


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

        # Warm up DeepFace by running a dummy analyze
        try:
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

                lm_expression: LandmarkExpression | None = None
                face_area_val: float | None = None

                if face_detected:
                    lmarks = results.face_landmarks[0]
                    ih, iw = frame.shape[:2]

                    # Head pose via solvePnP on 6 key landmarks
                    head_pose = _estimate_head_pose(lmarks, iw, ih)
                    movement = _head_movement(prev_pose, head_pose)
                    prev_pose = head_pose

                    # Landmark expression features (smile, mouth, eyes)
                    lm_expression = _compute_landmark_expression(lmarks, iw, ih)

                    # Crop face from landmarks for emotion accuracy
                    xs = [lm.x * iw for lm in lmarks]
                    ys = [lm.y * ih for lm in lmarks]
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))

                    # Face area for lean-in/lean-back tracking (normalized by frame area)
                    frame_area = iw * ih
                    face_area_val = float((x2 - x1) * (y2 - y1)) / frame_area if frame_area > 0 else 0.0

                    margin = int(0.2 * (x2 - x1))
                    face_crop = frame[
                        max(0, y1 - margin):min(ih, y2 + margin),
                        max(0, x1 - margin):min(iw, x2 + margin),
                    ]
                    if face_crop.size == 0:
                        face_crop = frame

                    enhanced = _preprocess_frame(face_crop)

                    # DeepFace emotion classification
                    try:
                        df_results = DeepFace.analyze(
                            enhanced, actions=["emotion"],
                            enforce_detection=False, silent=True,
                            detector_backend="skip",
                        )
                        if df_results:
                            result = df_results[0] if isinstance(df_results, list) else df_results
                            raw_7class, collapsed = _deepface_to_emotions(result["emotion"])
                            # Compute confidence before smoothing to drive adaptive α
                            face_conf = emotion_confidence(collapsed)
                            collapsed_emotions = _smooth_emotions(
                                collapsed, self._smoothed_emotions,
                                confidence=face_conf,
                            )
                            self._smoothed_emotions = collapsed_emotions
                            face_score = _engagement_score(
                                collapsed_emotions,
                                landmark_expr=lm_expression,
                                movement=movement,
                                head_pose=head_pose,
                            )
                            raw_emotions = raw_7class
                            dominant_emotion = max(raw_7class, key=raw_7class.get)  # type: ignore[arg-type]
                    except Exception:
                        pass
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
                    landmark_expression=lm_expression,
                    face_area=face_area_val,
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
