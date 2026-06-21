"""Reactor: unified reaction pipeline for ClaudeDJ.

Combines webcam frames and CLI feedback into a single reaction stream.
Produces windowed engagement scores on demand for the agent's decision loop.
Implements FR-6 (windowing), FR-7 (context-conditioned interpretation),
and FR-8 (CLI as first-class signal).
"""

from __future__ import annotations

import threading
import time
from collections import deque

from reaction import (
    Baseline,
    ReactionFrame,
    ReactionScore,
    Sentiment,
    SignalSource,
    TrackContext,
    aggregate_window,
    capture_baseline,
    cli_to_reaction_score,
)
from webcam import WebcamWorker


class Reactor:
    """Unified reaction pipeline.

    Manages the webcam worker, accepts CLI feedback, and produces
    windowed engagement scores for the agent.

    Usage:
        reactor = Reactor()
        reactor.start()

        # CLI feedback from user
        reactor.add_cli_feedback("like")

        # Agent asks for current engagement
        score = reactor.get_current_score(window_seconds=15)

        # Agent asks for recent trend
        trend = reactor.get_trend(windows=3, window_seconds=10)

        reactor.stop()
    """

    def __init__(
        self,
        camera_index: int = 0,
        enable_webcam: bool = True,
        history_size: int = 300,
    ):
        self._enable_webcam = enable_webcam
        self._webcam: WebcamWorker | None = None
        if enable_webcam:
            self._webcam = WebcamWorker(camera_index=camera_index)

        self._cli_scores: deque[ReactionScore] = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._track_context: TrackContext | None = None

    @property
    def baseline(self) -> Baseline | None:
        if self._webcam:
            return self._webcam.baseline
        return None

    @property
    def error(self) -> str | None:
        if self._webcam:
            return self._webcam.error
        return None

    def start(self) -> None:
        if self._webcam:
            self._webcam.start()

    def stop(self) -> None:
        if self._webcam:
            self._webcam.stop()

    def add_cli_feedback(self, feedback: str) -> ReactionScore:
        """Accept CLI feedback and return the resulting score."""
        score = cli_to_reaction_score(feedback)
        with self._lock:
            self._cli_scores.append(score)
        return score

    def set_track_context(
        self, energy: float = 0.5, valence: float = 0.5, cluster: str | None = None,
    ) -> None:
        """Update the current track context for context-conditioned scoring (FR-7)."""
        self._track_context = TrackContext(energy=energy, valence=valence, cluster=cluster)

    def get_current_score(self, window_seconds: float = 15.0) -> ReactionScore:
        """Get the current engagement score over the last N seconds (FR-6).

        Blends webcam frames and CLI feedback. CLI feedback overrides
        webcam signals when present in the window (FR-8: CLI is first-class).
        """
        now = time.time()
        cutoff = now - window_seconds

        # Check for recent CLI feedback first — it's highest confidence
        with self._lock:
            recent_cli = [s for s in self._cli_scores if s.window_end >= cutoff]

        if recent_cli:
            # Use the most recent CLI feedback as the primary signal
            return recent_cli[-1]

        # Fall back to webcam aggregation
        if self._webcam:
            baseline = self._webcam.baseline or Baseline()
            frames = [
                f for f in self._webcam.get_recent_frames(n=30)
                if f.timestamp >= cutoff
            ]
            if frames:
                return aggregate_window(frames, baseline, track_context=self._track_context)

        # No data at all
        return ReactionScore(
            score=0.5,
            confidence=0.0,
            sentiment=Sentiment.NEUTRAL,
            window_start=cutoff,
            window_end=now,
            frame_count=0,
        )

    def get_trend(
        self, windows: int = 3, window_seconds: float = 10.0
    ) -> list[ReactionScore]:
        """Get a trend of scores over the last N windows.

        Returns oldest-first. Useful for the agent to see if engagement
        is rising, falling, or stable.
        """
        now = time.time()
        scores: list[ReactionScore] = []

        # Fetch all frames once instead of copying per-window
        if self._webcam:
            all_frames = self._webcam.get_all_frames()
            baseline = self._webcam.baseline or Baseline()
        else:
            all_frames = []
            baseline = Baseline()

        for i in range(windows - 1, -1, -1):
            w_end = now - (i * window_seconds)
            w_start = w_end - window_seconds

            # Check CLI in this window
            with self._lock:
                cli_in_window = [
                    s for s in self._cli_scores
                    if w_start <= s.window_end <= w_end
                ]

            if cli_in_window:
                scores.append(cli_in_window[-1])
                continue

            # Filter pre-fetched webcam frames for this window
            if self._webcam:
                frames = [
                    f for f in all_frames
                    if w_start <= f.timestamp <= w_end
                ]
                scores.append(aggregate_window(frames, baseline, track_context=self._track_context))
            else:
                scores.append(ReactionScore(
                    score=0.5,
                    confidence=0.0,
                    sentiment=Sentiment.NEUTRAL,
                    window_start=w_start,
                    window_end=w_end,
                ))

        return scores

    def get_summary(self) -> dict:
        """Compact summary for get_session_context (FR-20).

        Returns the shape the agent needs for its decision bundle.
        """
        current = self.get_current_score()
        trend = self.get_trend()

        trend_direction = "stable"
        if len(trend) >= 2:
            delta = trend[-1].score - trend[0].score
            if delta > 0.15:
                trend_direction = "rising"
            elif delta < -0.15:
                trend_direction = "falling"

        # Get latest emotions, head pose, and landmark expression from webcam frames
        latest_emotions = None
        raw_emotions = None
        dominant_emotion = None
        head_pose = None
        landmark_expression = None
        if self._webcam:
            recent = self._webcam.get_recent_frames(n=1)
            if recent:
                frame = recent[-1]
                if frame.emotions:
                    latest_emotions = {k: float(v) for k, v in frame.emotions.items()}
                if frame.raw_emotions:
                    raw_emotions = {k: float(v) for k, v in frame.raw_emotions.items()}
                    dominant_emotion = frame.dominant_emotion
                if frame.head_pose:
                    head_pose = {
                        "yaw": float(frame.head_pose.yaw),
                        "pitch": float(frame.head_pose.pitch),
                        "roll": float(frame.head_pose.roll),
                    }
                if frame.landmark_expression:
                    landmark_expression = {
                        "smile": float(frame.landmark_expression.smile),
                        "mouth_open": float(frame.landmark_expression.mouth_open),
                        "ear": float(frame.landmark_expression.ear),
                        "brow_height": float(frame.landmark_expression.brow_height),
                    }

        return {
            "current_score": current.score,
            "confidence": current.confidence,
            "sentiment": current.sentiment.value,
            "trend_direction": trend_direction,
            "trend_scores": [round(s.score, 2) for s in trend],
            "source": current.source.value,
            "emotions": latest_emotions,
            "raw_emotions": raw_emotions,
            "dominant_emotion": dominant_emotion,
            "head_pose": head_pose,
            "landmark_expression": landmark_expression,
        }
