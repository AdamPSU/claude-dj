"""Tests for thread-safe SystemState."""

import time
import threading
import numpy as np
from vibe_dj.state import SystemState


class TestSystemState:
    def test_initial_valence_is_neutral(self):
        s = SystemState()
        assert s.valence == 0.5
        assert s.emotion_bucket == "neutral"

    def test_update_pose_is_readable(self):
        s = SystemState()
        s.update_pose(pitch=10.0, yaw=-5.0, roll=2.0, face_scale=1.2, face_detected=True)
        snap = s.snapshot()
        assert snap["pitch"] == 10.0
        assert snap["face_detected"] is True

    def test_pitch_buffer_respects_window(self):
        s = SystemState()
        now = time.time()
        s.append_pitch(now - 5.0, 1.0)  # old
        s.append_pitch(now - 1.0, 2.0)  # recent
        s.append_pitch(now, 3.0)         # now
        window = s.get_pitch_window(2.0)
        assert len(window) == 2
        assert window[0][1] == 2.0
        assert window[1][1] == 3.0

    def test_face_crop_copy_on_write(self):
        s = SystemState()
        crop = np.ones((48, 48, 3), dtype=np.uint8)
        s.set_face_crop(crop)
        crop[:] = 0  # mutate original
        retrieved = s.get_face_crop()
        assert retrieved is not None
        assert retrieved[0, 0, 0] == 1  # copy is unaffected

    def test_concurrent_writes_dont_corrupt(self):
        s = SystemState()

        def writer_pose():
            for i in range(100):
                s.update_pose(float(i), 0.0, 0.0, 1.0, True)

        def writer_emotion():
            for i in range(100):
                s.update_emotion(float(i) / 100, "positive",
                                 {"positive": 1.0, "neutral": 0.0, "negative": 0.0})

        t1 = threading.Thread(target=writer_pose)
        t2 = threading.Thread(target=writer_emotion)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        snap = s.snapshot()
        assert isinstance(snap["pitch"], float)
        assert isinstance(snap["valence"], float)
