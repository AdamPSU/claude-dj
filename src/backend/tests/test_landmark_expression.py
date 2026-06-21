import unittest

from claude_dj.reactions.reaction import LandmarkExpression, ReactionFrame
from claude_dj.reactions.webcam import _compute_ear, _compute_landmark_expression, _compute_mouth_ratio, _compute_smile


class FakeLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


IDX = {
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
    "left_brow_a": 70,
    "left_brow_b": 63,
    "left_brow_c": 105,
    "right_brow_a": 300,
    "right_brow_b": 293,
    "right_brow_c": 334,
}

IMG_W = 640
IMG_H = 480


def make_landmarks(**overrides: tuple[float, float]) -> list[FakeLandmark]:
    defaults = {
        "upper_lip": (0.50, 0.60),
        "lower_lip": (0.50, 0.63),
        "left_mouth": (0.42, 0.62),
        "right_mouth": (0.58, 0.62),
        "left_eye_outer": (0.34, 0.37),
        "left_eye_inner": (0.43, 0.37),
        "left_eye_upper1": (0.38, 0.34),
        "left_eye_lower1": (0.38, 0.38),
        "left_eye_upper2": (0.40, 0.34),
        "left_eye_lower2": (0.40, 0.38),
        "right_eye_outer": (0.66, 0.37),
        "right_eye_inner": (0.57, 0.37),
        "right_eye_upper1": (0.62, 0.34),
        "right_eye_lower1": (0.62, 0.38),
        "right_eye_upper2": (0.60, 0.34),
        "right_eye_lower2": (0.60, 0.38),
        "left_brow_a": (0.36, 0.30),
        "left_brow_b": (0.39, 0.30),
        "left_brow_c": (0.42, 0.30),
        "right_brow_a": (0.58, 0.30),
        "right_brow_b": (0.61, 0.30),
        "right_brow_c": (0.64, 0.30),
    }
    defaults.update(overrides)
    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(478)]
    for name, (x, y) in defaults.items():
        landmarks[IDX[name]] = FakeLandmark(x, y)
    return landmarks


class LandmarkExpressionTests(unittest.TestCase):
    def test_smiling_face_has_higher_smile_score_than_neutral_face(self) -> None:
        neutral = make_landmarks()
        smiling = make_landmarks(left_mouth=(0.38, 0.58), right_mouth=(0.62, 0.58))

        self.assertGreater(_compute_smile(smiling, IMG_W, IMG_H), _compute_smile(neutral, IMG_W, IMG_H))

    def test_open_mouth_has_higher_ratio_than_closed_mouth(self) -> None:
        closed = make_landmarks(upper_lip=(0.50, 0.61), lower_lip=(0.50, 0.62))
        open_mouth = make_landmarks(upper_lip=(0.50, 0.58), lower_lip=(0.50, 0.70))

        self.assertGreater(_compute_mouth_ratio(open_mouth, IMG_W, IMG_H), _compute_mouth_ratio(closed, IMG_W, IMG_H))

    def test_closed_eyes_have_lower_ear_than_open_eyes(self) -> None:
        open_eyes = make_landmarks()
        closed_eyes = make_landmarks(
            left_eye_upper1=(0.38, 0.37),
            left_eye_lower1=(0.38, 0.37),
            left_eye_upper2=(0.40, 0.37),
            left_eye_lower2=(0.40, 0.37),
            right_eye_upper1=(0.62, 0.37),
            right_eye_lower1=(0.62, 0.37),
            right_eye_upper2=(0.60, 0.37),
            right_eye_lower2=(0.60, 0.37),
        )

        self.assertLess(_compute_ear(closed_eyes, IMG_W, IMG_H), _compute_ear(open_eyes, IMG_W, IMG_H))

    def test_combined_compute_returns_landmark_expression(self) -> None:
        expression = _compute_landmark_expression(make_landmarks(), IMG_W, IMG_H)

        self.assertIsInstance(expression, LandmarkExpression)

    def test_reaction_frame_accepts_landmark_expression(self) -> None:
        expression = LandmarkExpression(smile=0.6, mouth_open=0.2, ear=0.3)

        frame = ReactionFrame(landmark_expression=expression)

        self.assertEqual(frame.landmark_expression, expression)


if __name__ == "__main__":
    unittest.main()
