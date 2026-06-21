from __future__ import annotations

import threading
import time
from collections import deque
from typing import Protocol

from .models import Baseline, ReactionFrame, ReactionScore, Sentiment, TrackContext
from .scoring import aggregate_window, cli_to_reaction_score


class SummaryProvider(Protocol):
    def get_summary(self) -> dict[str, object]: ...


class FrameSource(Protocol):
    @property
    def baseline(self) -> Baseline | None: ...

    @property
    def error(self) -> str | None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get_recent_frames(self, n: int = 10) -> list[ReactionFrame]: ...

    def get_all_frames(self) -> list[ReactionFrame]: ...


class Reactor:
    def __init__(self, frame_source: FrameSource | None = None, history_size: int = 300) -> None:
        self.frame_source = frame_source
        self._cli_scores: deque[ReactionScore] = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._track_context: TrackContext | None = None

    @property
    def error(self) -> str | None:
        return self.frame_source.error if self.frame_source else None

    def start(self) -> None:
        if self.frame_source:
            self.frame_source.start()

    def stop(self) -> None:
        if self.frame_source:
            self.frame_source.stop()

    def add_cli_feedback(self, feedback: str) -> ReactionScore:
        score = cli_to_reaction_score(feedback)
        with self._lock:
            self._cli_scores.append(score)
        return score

    def set_track_context(self, *, energy: float = 0.5, cluster: str | None = None) -> None:
        self._track_context = TrackContext(energy=energy, cluster=cluster)

    def get_current_score(self, window_seconds: float = 15.0) -> ReactionScore:
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            recent_cli = [score for score in self._cli_scores if score.window_end >= cutoff]
        if recent_cli:
            return recent_cli[-1]
        if self.frame_source:
            baseline = self.frame_source.baseline or Baseline()
            frames = [frame for frame in self.frame_source.get_recent_frames(n=30) if frame.timestamp >= cutoff]
            if frames:
                return aggregate_window(frames, baseline, track_context=self._track_context)
        return ReactionScore(
            score=0.5,
            confidence=0.0,
            sentiment=Sentiment.NEUTRAL,
            window_start=cutoff,
            window_end=now,
            frame_count=0,
        )

    def get_trend(self, *, windows: int = 3, window_seconds: float = 10.0) -> list[ReactionScore]:
        now = time.time()
        frames = self.frame_source.get_all_frames() if self.frame_source else []
        baseline = self.frame_source.baseline if self.frame_source and self.frame_source.baseline else Baseline()
        scores: list[ReactionScore] = []
        for index in range(windows - 1, -1, -1):
            window_end = now - (index * window_seconds)
            window_start = window_end - window_seconds
            with self._lock:
                cli_scores = [score for score in self._cli_scores if window_start <= score.window_end <= window_end]
            if cli_scores:
                scores.append(cli_scores[-1])
                continue
            window_frames = [frame for frame in frames if window_start <= frame.timestamp <= window_end]
            scores.append(aggregate_window(window_frames, baseline, track_context=self._track_context))
        return scores

    def get_summary(self) -> dict[str, object]:
        current = self.get_current_score()
        trend = self.get_trend()
        trend_direction = "stable"
        if len(trend) >= 2:
            delta = trend[-1].score - trend[0].score
            if delta > 0.15:
                trend_direction = "rising"
            elif delta < -0.15:
                trend_direction = "falling"

        latest_frame = None
        if self.frame_source:
            recent = self.frame_source.get_recent_frames(n=1)
            latest_frame = recent[-1] if recent else None

        return {
            "current_score": current.score,
            "confidence": current.confidence,
            "sentiment": current.sentiment.value,
            "trend_direction": trend_direction,
            "trend_scores": [round(score.score, 2) for score in trend],
            "source": current.source.value,
            "emotions": latest_frame.emotions if latest_frame else None,
            "emotion_probs": latest_frame.emotion_probs if latest_frame else None,
            "emotion_bucket": latest_frame.emotion_bucket if latest_frame else None,
            "valence": latest_frame.valence if latest_frame else None,
            "raw_emotions": latest_frame.raw_emotions if latest_frame else None,
            "dominant_emotion": latest_frame.dominant_emotion if latest_frame else None,
            "face_scale": latest_frame.face_scale if latest_frame else None,
            "vibe_score": latest_frame.vibe_score if latest_frame else None,
            "plv": latest_frame.plv if latest_frame else None,
            "period_match_score": latest_frame.period_match_score if latest_frame else None,
            "motion_energy": latest_frame.movement if latest_frame else None,
            "head_pose": {
                "yaw": latest_frame.head_pose.yaw,
                "pitch": latest_frame.head_pose.pitch,
                "roll": latest_frame.head_pose.roll,
            }
            if latest_frame and latest_frame.head_pose
            else None,
        }


class ReactorReactionSource:
    def __init__(self, reactor: SummaryProvider) -> None:
        self.reactor = reactor

    async def get_reaction_signal(self) -> dict[str, object]:
        summary = self.reactor.get_summary()
        score = self._float_value(summary.get("current_score"))
        confidence = self._float_value(summary.get("confidence"))
        sentiment = str(summary.get("sentiment") or "neutral")
        trend_direction = str(summary.get("trend_direction") or "stable")
        source = str(summary.get("source") or "reactor")
        signed_score = round((score * 2.0) - 1.0, 3)
        return {
            "available": confidence > 0.0,
            "stub": False,
            "source": source,
            "trend": sentiment,
            "confidence": confidence,
            "score": signed_score,
            "camera_state": trend_direction,
            "summary": f"{sentiment} reaction, confidence {confidence:.2f}, trend {trend_direction}.",
            "raw": summary,
        }

    @staticmethod
    def _float_value(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
