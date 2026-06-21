import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from claude_dj.reactions.hud import build_hud_snapshot, draw_vibedj_hud
from claude_dj.reactions.reaction import HeadPose, ReactionFrame
from claude_dj.reactions.webcam import WebcamWorker


class FakeFrame:
    def copy(self):
        return self


class VibeDJHudTests(unittest.TestCase):
    def test_webcam_preview_defaults_to_vibedj_window(self) -> None:
        worker = WebcamWorker()

        self.assertEqual(worker.preview_window_name, "VibeDJ")

    def test_preview_cadence_is_separate_from_reaction_sampling(self) -> None:
        worker = WebcamWorker()

        self.assertEqual(worker.sample_interval, 1.0)
        self.assertEqual(worker.preview_fps, 30.0)

    def test_preview_frame_can_update_without_new_reaction_frame(self) -> None:
        worker = WebcamWorker(show_preview=True)
        frame = ReactionFrame(presence=1.0, head_pose=HeadPose(pitch=5.0))

        worker._store_preview_frame(FakeFrame(), frame)
        worker._store_preview_frame(FakeFrame(), None)

        self.assertIs(worker._preview_reaction_frame, frame)
        self.assertEqual(list(worker._pitch_history), [5.0])

    def test_negative_reaction_snapshot_displays_change_track_action(self) -> None:
        frame = ReactionFrame(
            presence=1.0,
            face=0.05,
            movement=1.5,
            head_pose=HeadPose(pitch=-12.0),
            emotions={"happy": 0.0, "neutral": 0.1, "disinterested": 0.9},
        )

        snapshot = build_hud_snapshot(frame)

        self.assertEqual(snapshot["emotion_bucket"], "negative")
        self.assertEqual(snapshot["agent_action"], "change_track")
        self.assertTrue(snapshot["face_detected"])
        self.assertGreater(snapshot["motion_energy"], 0.0)

    def test_preview_pump_renders_vibedj_hud(self) -> None:
        displayed: list[tuple[str, object]] = []
        fake_cv2 = SimpleNamespace(
            imshow=lambda name, frame: displayed.append((name, frame)),
            waitKey=lambda delay: -1,
        )
        worker = WebcamWorker(show_preview=True)
        with worker._lock:
            worker._preview_frame = FakeFrame()
            worker._preview_reaction_frame = ReactionFrame(presence=1.0, head_pose=HeadPose(pitch=3.0))

        with patch.dict(sys.modules, {"cv2": fake_cv2}), patch(
            "claude_dj.reactions.webcam.draw_vibedj_hud",
            side_effect=lambda frame, reaction_frame, pitch_history: frame,
        ) as draw_hud:
            handled_quit = worker.pump_preview_window()

        self.assertFalse(handled_quit)
        draw_hud.assert_called_once()
        self.assertEqual(displayed, [("VibeDJ", worker._preview_frame)])

    def test_draw_vibedj_hud_updates_frame_pixels(self) -> None:
        try:
            import cv2  # noqa: F401
            import numpy as np
        except ImportError as exc:
            self.skipTest(f"OpenCV/numpy unavailable: {exc}")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        reaction_frame = ReactionFrame(presence=1.0, face=0.9, head_pose=HeadPose(pitch=4.0))

        result = draw_vibedj_hud(frame, reaction_frame, [0.0, 4.0])

        self.assertIs(result, frame)
        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
