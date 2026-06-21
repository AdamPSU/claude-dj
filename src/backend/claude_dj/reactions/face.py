from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import config


@dataclass(frozen=True)
class FaceResult:
    pitch: float
    yaw: float
    roll: float
    face_scale: float
    face_crop: Any
    landmarks: list[Any]


FACE_3D = np.array(
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
LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]
LEFT_EYE_INDEX = 33
RIGHT_EYE_INDEX = 263


def find_face_model(model_path: str | os.PathLike[str] | None = None) -> str:
    candidates = []
    if model_path is not None:
        candidates.append(Path(model_path))
    candidates.append(Path(__file__).resolve().parents[2] / "face_landmarker.task")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(
        "face_landmarker.task not found. Download it from "
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )


class FaceProcessor:
    def __init__(self, *, model_path: str | os.PathLike[str] | None = None) -> None:
        import mediapipe as mp

        self._cv2 = __import__("cv2")
        self._mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=find_face_model(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=config.FACE_MESH_MAX_FACES,
            min_face_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process(self, bgr_frame: Any) -> FaceResult | None:
        cv2 = self._cv2
        mp = self._mp
        height, width = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(image)
        if not results.face_landmarks:
            return None

        landmarks = results.face_landmarks[0]
        left_eye = landmarks[LEFT_EYE_INDEX]
        right_eye = landmarks[RIGHT_EYE_INDEX]
        ipd_px = (((left_eye.x - right_eye.x) * width) ** 2 + ((left_eye.y - right_eye.y) * height) ** 2) ** 0.5
        face_scale = max(ipd_px / config.REFERENCE_IPD_PX, 0.1) if config.REFERENCE_IPD_PX > 0 else 1.0

        image_points = np.array(
            [(landmarks[index].x * width, landmarks[index].y * height) for index in LANDMARK_INDICES],
            dtype=np.float64,
        )
        camera_matrix = np.array(
            [[float(width), 0, width / 2], [0, float(width), height / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        ok, rotation_vector, _ = cv2.solvePnP(
            FACE_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        nose_2d, _ = cv2.projectPoints(
            np.array([[0.0, 0.0, 1000.0]], dtype=np.float64).reshape(1, 1, 3),
            rotation_vector,
            np.zeros((3, 1)),
            camera_matrix,
            dist_coeffs,
        )
        nose_tip = image_points[0]
        projected = (float(nose_2d[0, 0, 0]), float(nose_2d[0, 0, 1]))
        raw_yaw = (projected[0] - nose_tip[0]) / width * 90.0
        raw_pitch = (projected[1] - nose_tip[1]) / height * 90.0
        roll = float(
            np.degrees(
                np.arctan2(
                    image_points[3][1] - image_points[2][1],
                    image_points[3][0] - image_points[2][0],
                )
            )
        )
        pitch = raw_pitch / face_scale

        xs = [landmark.x * width for landmark in landmarks]
        ys = [landmark.y * height for landmark in landmarks]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        margin = int(0.2 * max(1, x2 - x1))
        crop = bgr_frame[max(0, y1 - margin) : min(height, y2 + margin), max(0, x1 - margin) : min(width, x2 + margin)]
        if getattr(crop, "size", 0) == 0:
            crop = bgr_frame

        return FaceResult(
            pitch=round(float(np.clip(pitch, -90, 90)), 2),
            yaw=round(float(np.clip(raw_yaw, -90, 90)), 2),
            roll=round(float(np.clip(roll, -90, 90)), 2),
            face_scale=round(face_scale, 3),
            face_crop=crop,
            landmarks=landmarks,
        )

    def close(self) -> None:
        self._landmarker.close()
