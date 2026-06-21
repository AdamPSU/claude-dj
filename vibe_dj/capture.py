"""Webcam capture thread.

Grabs frames from the webcam at the target FPS and passes them to
a callback. Runs in a daemon thread so it never blocks the main loop.
"""

from __future__ import annotations

import threading
import time

import cv2

from vibe_dj import config


class CaptureThread:
    """Background thread that reads webcam frames."""

    def __init__(self, on_frame, camera_index: int = config.CAMERA_INDEX):
        self._on_frame = on_frame
        self._camera_index = camera_index
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            print("[capture] ERROR: could not open webcam")
            self._running = False
            return

        interval = 1.0 / config.FPS_TARGET
        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(interval)
                continue
            self._on_frame(frame)
            time.sleep(max(0, interval - 0.005))

        cap.release()
