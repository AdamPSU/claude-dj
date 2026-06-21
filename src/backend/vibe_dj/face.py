"""FaceMesh: head pose (pitch/yaw/roll) + face crop, scale-normalized.

Uses ONE MediaPipe FaceLandmarker pass per frame for both head pose and
the face region crop (used downstream by emotion.py).

Head pose is estimated via cv2.solvePnP on 6 canonical landmarks.
Pitch is scale-normalized by the inter-pupillary distance so that
camera distance does not change the apparent bob amplitude.
"""

from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np

from vibe_dj import config

# MediaPipe Tasks API (mp.solutions is unavailable in this version)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model path — look next to this file first, then in src/backend/
_MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "face_landmarker.task"),
    os.path.join(os.path.dirname(__file__), "..", "src", "backend", "face_landmarker.task"),
]


def _find_model() -> str:
    for p in _MODEL_CANDIDATES:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "face_landmarker.task not found. Download it with:\n"
        "curl -L -o face_landmarker.task "
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )


# 3D model points for a canonical face (mm).
# Nose tip (1), chin (199), left eye outer (33),
# right eye outer (263), left mouth (61), right mouth (291).
_FACE_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

_LM_IDX = [1, 199, 33, 263, 61, 291]

# IPD landmarks (left/right eye outer corners)
_LEFT_EYE_IDX = 33
_RIGHT_EYE_IDX = 263


class FaceResult:
    """Output of one FaceLandmarker pass."""

    __slots__ = ("pitch", "yaw", "roll", "face_scale", "face_crop", "landmarks")

    def __init__(
        self, pitch: float, yaw: float, roll: float,
        face_scale: float, face_crop: np.ndarray,
        landmarks: list,
    ):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.face_scale = face_scale
        self.face_crop = face_crop
        self.landmarks = landmarks


class FaceProcessor:
    """Wraps MediaPipe FaceLandmarker + solvePnP head-pose estimation."""

    def __init__(self):
        model_path = _find_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=config.FACE_MESH_MAX_FACES,
            min_face_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def process(self, bgr_frame: np.ndarray) -> FaceResult | None:
        """Run FaceLandmarker on a BGR frame.

        Returns FaceResult with scale-normalized pitch, or None if no face.
        """
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        lms = results.face_landmarks[0]

        # --- Inter-pupillary distance for scale normalization ---
        le = lms[_LEFT_EYE_IDX]
        re = lms[_RIGHT_EYE_IDX]
        ipd_px = ((le.x - re.x) * w) ** 2 + ((le.y - re.y) * h) ** 2
        ipd_px = ipd_px ** 0.5
        face_scale = ipd_px / config.REFERENCE_IPD_PX if config.REFERENCE_IPD_PX > 0 else 1.0
        face_scale = max(face_scale, 0.1)  # avoid division by tiny values

        # --- solvePnP for head pose ---
        image_points = np.array(
            [(lms[i].x * w, lms[i].y * h) for i in _LM_IDX],
            dtype=np.float64,
        )
        focal_length = float(w)
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, _ = cv2.solvePnP(
            _FACE_3D, image_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        # Project a point along the nose axis to derive pitch/yaw
        nose_3d = np.array([[0.0, 0.0, 1000.0]], dtype=np.float64)
        nose_2d, _ = cv2.projectPoints(
            nose_3d.reshape(1, 1, 3), rvec, np.zeros((3, 1)),
            cam_matrix, dist_coeffs,
        )
        p1 = image_points[0]  # nose tip
        p2 = (float(nose_2d[0, 0, 0]), float(nose_2d[0, 0, 1]))

        raw_yaw = (p2[0] - p1[0]) / w * 90.0
        raw_pitch = (p2[1] - p1[1]) / h * 90.0

        # Roll from eye line
        roll = float(np.degrees(np.arctan2(
            image_points[3][1] - image_points[2][1],
            image_points[3][0] - image_points[2][0],
        )))

        # Scale-normalize pitch so distance from camera doesn't change
        # the apparent amplitude of head bobs.
        pitch = raw_pitch / face_scale
        yaw = float(np.clip(raw_yaw, -90, 90))
        pitch = float(np.clip(pitch, -90, 90))
        roll = float(np.clip(roll, -90, 90))

        # --- Face crop from landmark bounding box ---
        xs = [lm.x * w for lm in lms]
        ys = [lm.y * h for lm in lms]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        margin = int(0.2 * (x2 - x1))
        crop = bgr_frame[
            max(0, y1 - margin): min(h, y2 + margin),
            max(0, x1 - margin): min(w, x2 + margin),
        ]
        if crop.size == 0:
            crop = bgr_frame  # fallback

        return FaceResult(
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            face_scale=round(face_scale, 3),
            face_crop=crop,
            landmarks=lms,
        )

    def close(self) -> None:
        self._landmarker.close()
