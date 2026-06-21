from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from . import config
from .emotion import DeepFaceClassifier, ema_smooth, to_valence
from .face import FaceProcessor
from .models import Baseline, HeadPose, ReactionFrame, SignalSource
from .scoring import capture_baseline, emotion_confidence
from .vibe import VibeDetector, generated_beat_times


DEFAULT_FACE_MODEL_PATH = Path(__file__).resolve().parents[2] / "face_landmarker.task"


class WebcamWorker:
    def __init__(
        self,
        *,
        camera_index: int = 0,
        sample_interval: float = 1.0 / config.FPS_TARGET,
        buffer_size: int = 300,
        baseline_frames: int = 30,
        model_path: str | os.PathLike[str] = DEFAULT_FACE_MODEL_PATH,
        show_preview: bool = False,
        preview_window_name: str = "ClaudeDJ Emotion Detection",
        bpm: float = config.DEFAULT_BPM,
    ) -> None:
        self.camera_index = camera_index
        self.sample_interval = sample_interval
        self.buffer_size = buffer_size
        self.baseline_frames = baseline_frames
        self.model_path = str(model_path)
        self.show_preview = show_preview
        self.preview_window_name = preview_window_name
        self.bpm = bpm
        self._frames: deque[ReactionFrame] = deque(maxlen=buffer_size)
        self._pitch_buffer: deque[tuple[float, float]] = deque(maxlen=buffer_size)
        self._baseline: Baseline | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._previous_pitch: float | None = None
        self._emotion_probs: dict[str, float] | None = None
        self._emotion_bucket = "neutral"
        self._valence = 0.5
        self._frame_count = 0
        self._preview_frame: Any | None = None
        self._vibe = VibeDetector()

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
        cv2, _, _, _ = self._optional_modules()
        classifier = DeepFaceClassifier()
        face_processor = FaceProcessor(model_path=self.model_path)
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self._error = f"Could not open webcam index {self.camera_index}"
            self._running = False
            face_processor.close()
            return

        baseline_buffer: list[ReactionFrame] = []
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(self.sample_interval)
                    continue
                reaction_frame = self._frame_to_reaction(frame, classifier, face_processor)
                with self._lock:
                    self._frames.append(reaction_frame)
                if self._baseline is None:
                    baseline_buffer.append(reaction_frame)
                    if len(baseline_buffer) >= self.baseline_frames:
                        self._baseline = capture_baseline(baseline_buffer)
                if self.show_preview:
                    self._store_preview_frame(frame)
                time.sleep(self.sample_interval)
        finally:
            cap.release()
            face_processor.close()

    def _frame_to_reaction(self, frame: Any, classifier: DeepFaceClassifier, face_processor: FaceProcessor) -> ReactionFrame:
        now = time.time()
        face = face_processor.process(frame)
        if face is None:
            self._previous_pitch = None
            reaction_frame = ReactionFrame(timestamp=now, presence=0.0, source=SignalSource.WEBCAM)
            if self.show_preview:
                self._draw_preview_overlay(frame, reaction_frame)
            return reaction_frame

        movement = 0.0
        if self._previous_pitch is not None:
            movement = round(min(1.0, abs(face.pitch - self._previous_pitch) / 15.0), 3)
        self._previous_pitch = face.pitch
        self._frame_count += 1

        with self._lock:
            self._pitch_buffer.append((now, face.pitch))
            pitch_window = [(timestamp, pitch) for timestamp, pitch in self._pitch_buffer if timestamp >= now - config.WINDOW_LEN_S]

        if self._frame_count % config.EMOTION_CADENCE == 0 or self._emotion_probs is None:
            current_probs = classifier.classify(face.face_crop)
            self._emotion_probs = ema_smooth(current_probs, self._emotion_probs)
            self._valence = round(max(0.0, min(1.0, to_valence(self._emotion_probs))), 3)
            self._emotion_bucket = max(self._emotion_probs, key=self._emotion_probs.get)

        beat_times = generated_beat_times(pitch_window[0][0], pitch_window[-1][0], self.bpm) if pitch_window else []
        vibe_score, plv, period_match = self._vibe.compute(pitch_window, beat_times, self.bpm)
        emotion_probs = dict(self._emotion_probs or {"positive": 0.0, "neutral": 1.0, "negative": 0.0})
        reaction_frame = ReactionFrame(
            timestamp=now,
            presence=1.0,
            movement=movement,
            head_pose=HeadPose(yaw=face.yaw, pitch=face.pitch, roll=face.roll),
            face_scale=face.face_scale,
            face=self._valence,
            emotions=emotion_probs,
            emotion_probs=emotion_probs,
            emotion_bucket=self._emotion_bucket,
            valence=self._valence,
            dominant_emotion=self._emotion_bucket,
            emotion_confidence=emotion_confidence(emotion_probs),
            vibe_score=round(vibe_score, 3),
            plv=round(plv, 3),
            period_match_score=round(period_match, 3),
            source=SignalSource.WEBCAM,
        )
        if self.show_preview:
            self._draw_preview_overlay(frame, reaction_frame, landmarks=face.landmarks)
        return reaction_frame

    def _draw_preview_overlay(self, frame: Any, reaction_frame: ReactionFrame, *, landmarks: list[Any] | None = None) -> None:
        cv2, _, _, _ = self._optional_modules()
        if landmarks is not None:
            image_height, image_width = frame.shape[:2]
            for landmark in landmarks:
                center = (int(landmark.x * image_width), int(landmark.y * image_height))
                cv2.circle(frame, center, 1, (0, 255, 0), -1)

        status = (
            f"presence={float(reaction_frame.presence or 0.0):.0f} "
            f"valence={float(reaction_frame.valence or 0.5):.2f} "
            f"vibe={float(reaction_frame.vibe_score or 0.0):.2f}"
        )
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if reaction_frame.emotion_bucket:
            cv2.putText(
                frame,
                f"emotion: {reaction_frame.emotion_bucket}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
        if not reaction_frame.emotion_probs:
            return
        y_offset = 90
        for emotion, value in sorted(reaction_frame.emotion_probs.items(), key=lambda item: -item[1]):
            bar_len = int(value * 300)
            color = (0, 255, 0) if emotion == "positive" else (0, 150, 255) if emotion == "neutral" else (0, 0, 255)
            cv2.putText(frame, emotion[:3], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.rectangle(frame, (50, y_offset - 10), (50 + bar_len, y_offset), color, -1)
            cv2.putText(frame, f"{value * 100:.1f}%", (55 + bar_len, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 22

    def _store_preview_frame(self, frame: Any) -> None:
        with self._lock:
            self._preview_frame = frame.copy()

    def pump_preview_window(self) -> bool:
        if not self.show_preview:
            return False
        with self._lock:
            frame = self._preview_frame.copy() if self._preview_frame is not None else None
        if frame is None:
            return False
        import cv2

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
            from scipy.signal import butter  # noqa: F401
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
