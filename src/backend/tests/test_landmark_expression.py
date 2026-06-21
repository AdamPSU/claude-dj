"""Tests for landmark-based expression features.

Validates smile detection, mouth opening ratio, and eye aspect ratio
computed directly from MediaPipe face landmark geometry.
"""
import math
import pytest


# ---------------------------------------------------------------------------
# Helpers: synthetic landmark generation
# ---------------------------------------------------------------------------

class FakeLandmark:
    """Mimics MediaPipe NormalizedLandmark (x, y in 0-1 range)."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


# Landmark indices used by the expression features
_IDX = {
    "nose_tip": 1,
    "upper_lip": 13,
    "lower_lip": 14,
    "left_mouth": 61,
    "right_mouth": 291,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "left_eye_upper1": 159,
    "left_eye_lower1": 145,
    "left_eye_upper2": 160,
    "left_eye_lower2": 144,
    "right_eye_outer": 263,
    "right_eye_inner": 362,
    "right_eye_upper1": 386,
    "right_eye_lower1": 374,
    "right_eye_upper2": 387,
    "right_eye_lower2": 373,
}

# Need enough slots for the highest landmark index we use (386, 387, etc.)
_NUM_LANDMARKS = 478


def _make_landmarks(**overrides) -> list:
    """Build a full landmark list with sensible defaults for a neutral face.

    Pass keyword overrides matching _IDX keys to shift specific points.
    Values are (x, y) tuples in normalized 0-1 coordinates.
    """
    defaults = {
        "nose_tip":         (0.50, 0.45),
        "upper_lip":        (0.50, 0.60),
        "lower_lip":        (0.50, 0.63),
        "left_mouth":       (0.42, 0.62),
        "right_mouth":      (0.58, 0.62),
        "left_eye_outer":   (0.34, 0.37),
        "left_eye_inner":   (0.43, 0.37),
        "left_eye_upper1":  (0.38, 0.34),
        "left_eye_lower1":  (0.38, 0.38),
        "left_eye_upper2":  (0.40, 0.34),
        "left_eye_lower2":  (0.40, 0.38),
        "right_eye_outer":  (0.66, 0.37),
        "right_eye_inner":  (0.57, 0.37),
        "right_eye_upper1": (0.62, 0.34),
        "right_eye_lower1": (0.62, 0.38),
        "right_eye_upper2": (0.60, 0.34),
        "right_eye_lower2": (0.60, 0.38),
    }
    defaults.update(overrides)

    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(_NUM_LANDMARKS)]
    for name, (x, y) in defaults.items():
        landmarks[_IDX[name]] = FakeLandmark(x, y)
    return landmarks


IMG_W, IMG_H = 640, 480


# ---------------------------------------------------------------------------
# Smile detection tests
# ---------------------------------------------------------------------------

class TestComputeSmile:
    """Smile score from lip corner geometry relative to lip center."""

    def test_neutral_face_low_smile(self):
        from webcam import _compute_smile
        landmarks = _make_landmarks()
        score = _compute_smile(landmarks, IMG_W, IMG_H)
        assert 0.0 <= score <= 0.3, f"Neutral face should have low smile, got {score}"

    def test_smiling_face_high_smile(self):
        from webcam import _compute_smile
        # Smile: lip corners move UP (lower y) and OUT
        landmarks = _make_landmarks(
            left_mouth=(0.38, 0.58),   # up and out from (0.42, 0.62)
            right_mouth=(0.62, 0.58),  # up and out from (0.58, 0.62)
        )
        score = _compute_smile(landmarks, IMG_W, IMG_H)
        assert score >= 0.5, f"Smiling face should have high smile score, got {score}"

    def test_smile_score_in_range(self):
        from webcam import _compute_smile
        landmarks = _make_landmarks()
        score = _compute_smile(landmarks, IMG_W, IMG_H)
        assert 0.0 <= score <= 1.0

    def test_bigger_smile_higher_score(self):
        from webcam import _compute_smile
        # Small smile
        small = _make_landmarks(
            left_mouth=(0.41, 0.60),
            right_mouth=(0.59, 0.60),
        )
        # Big smile
        big = _make_landmarks(
            left_mouth=(0.36, 0.56),
            right_mouth=(0.64, 0.56),
        )
        assert _compute_smile(big, IMG_W, IMG_H) > _compute_smile(small, IMG_W, IMG_H)


# ---------------------------------------------------------------------------
# Mouth opening tests
# ---------------------------------------------------------------------------

class TestComputeMouthRatio:
    """Mouth opening ratio from upper/lower lip distance."""

    def test_closed_mouth_low_ratio(self):
        from webcam import _compute_mouth_ratio
        # Lips close together
        landmarks = _make_landmarks(
            upper_lip=(0.50, 0.61),
            lower_lip=(0.50, 0.62),
        )
        ratio = _compute_mouth_ratio(landmarks, IMG_W, IMG_H)
        assert ratio <= 0.2, f"Closed mouth should have low ratio, got {ratio}"

    def test_open_mouth_high_ratio(self):
        from webcam import _compute_mouth_ratio
        # Mouth wide open (singing along)
        landmarks = _make_landmarks(
            upper_lip=(0.50, 0.58),
            lower_lip=(0.50, 0.70),
        )
        ratio = _compute_mouth_ratio(landmarks, IMG_W, IMG_H)
        assert ratio >= 0.5, f"Open mouth should have high ratio, got {ratio}"

    def test_ratio_in_range(self):
        from webcam import _compute_mouth_ratio
        landmarks = _make_landmarks()
        ratio = _compute_mouth_ratio(landmarks, IMG_W, IMG_H)
        assert 0.0 <= ratio <= 1.0

    def test_wider_opening_higher_ratio(self):
        from webcam import _compute_mouth_ratio
        slightly_open = _make_landmarks(
            upper_lip=(0.50, 0.59),
            lower_lip=(0.50, 0.65),
        )
        wide_open = _make_landmarks(
            upper_lip=(0.50, 0.57),
            lower_lip=(0.50, 0.72),
        )
        assert _compute_mouth_ratio(wide_open, IMG_W, IMG_H) > \
               _compute_mouth_ratio(slightly_open, IMG_W, IMG_H)


# ---------------------------------------------------------------------------
# Eye Aspect Ratio (EAR) tests
# ---------------------------------------------------------------------------

class TestComputeEAR:
    """Eye Aspect Ratio from eyelid landmark distances."""

    def test_open_eyes_moderate_ear(self):
        from webcam import _compute_ear
        landmarks = _make_landmarks()
        ear = _compute_ear(landmarks, IMG_W, IMG_H)
        assert 0.15 <= ear <= 0.6, f"Open eyes should have moderate EAR, got {ear}"

    def test_closed_eyes_low_ear(self):
        from webcam import _compute_ear
        # Upper and lower eyelids converge
        landmarks = _make_landmarks(
            left_eye_upper1=(0.38, 0.37),
            left_eye_lower1=(0.38, 0.37),
            left_eye_upper2=(0.40, 0.37),
            left_eye_lower2=(0.40, 0.37),
            right_eye_upper1=(0.62, 0.37),
            right_eye_lower1=(0.62, 0.37),
            right_eye_upper2=(0.60, 0.37),
            right_eye_lower2=(0.60, 0.37),
        )
        ear = _compute_ear(landmarks, IMG_W, IMG_H)
        assert ear <= 0.1, f"Closed eyes should have very low EAR, got {ear}"

    def test_wide_eyes_high_ear(self):
        from webcam import _compute_ear
        # Eyelids far apart
        landmarks = _make_landmarks(
            left_eye_upper1=(0.38, 0.31),
            left_eye_lower1=(0.38, 0.40),
            left_eye_upper2=(0.40, 0.31),
            left_eye_lower2=(0.40, 0.40),
            right_eye_upper1=(0.62, 0.31),
            right_eye_lower1=(0.62, 0.40),
            right_eye_upper2=(0.60, 0.31),
            right_eye_lower2=(0.60, 0.40),
        )
        ear = _compute_ear(landmarks, IMG_W, IMG_H)
        assert ear > 0.4, f"Wide eyes should have high EAR, got {ear}"

    def test_ear_in_range(self):
        from webcam import _compute_ear
        landmarks = _make_landmarks()
        ear = _compute_ear(landmarks, IMG_W, IMG_H)
        assert 0.0 <= ear <= 1.0


# ---------------------------------------------------------------------------
# Combined LandmarkExpression tests
# ---------------------------------------------------------------------------

class TestLandmarkExpression:
    """LandmarkExpression dataclass and combined compute function."""

    def test_dataclass_fields_exist(self):
        from reaction import LandmarkExpression
        expr = LandmarkExpression(smile=0.5, mouth_open=0.3, ear=0.25)
        assert expr.smile == 0.5
        assert expr.mouth_open == 0.3
        assert expr.ear == 0.25

    def test_compute_returns_landmark_expression(self):
        from webcam import _compute_landmark_expression
        from reaction import LandmarkExpression
        landmarks = _make_landmarks()
        result = _compute_landmark_expression(landmarks, IMG_W, IMG_H)
        assert isinstance(result, LandmarkExpression)

    def test_reaction_frame_has_landmark_expression(self):
        from reaction import ReactionFrame, LandmarkExpression
        expr = LandmarkExpression(smile=0.6, mouth_open=0.2, ear=0.3)
        frame = ReactionFrame(landmark_expression=expr)
        assert frame.landmark_expression is not None
        assert frame.landmark_expression.smile == 0.6
