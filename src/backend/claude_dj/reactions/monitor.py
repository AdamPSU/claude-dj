from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ReactionEvent:
    event_type: str
    current_track_id: str
    current_cluster: str | None
    duration_seconds: float
    signal: dict[str, Any]
    metadata: dict[str, Any] | None = None

    def to_prompt_data(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "current_track_id": self.current_track_id,
            "current_cluster": self.current_cluster,
            "duration_seconds": round(self.duration_seconds, 1),
            "signal": self.signal,
            "metadata": self.metadata or {},
        }


class ReactionMonitor:
    def __init__(
        self,
        *,
        negative_seconds: float = 5.0,
        confidence_threshold: float = 0.6,
        negative_score_threshold: float = 0.4,
        cooldown_seconds: float = 45.0,
        defer_after_progress: float = 0.75,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.negative_seconds = negative_seconds
        self.confidence_threshold = confidence_threshold
        self.negative_score_threshold = negative_score_threshold
        self.cooldown_seconds = cooldown_seconds
        self.defer_after_progress = defer_after_progress
        self.clock = clock
        self._negative_since: float | None = None
        self._cooldown_until = 0.0
        self._fired_track_id: str | None = None
        self._deferred_event: ReactionEvent | None = None

    async def poll(self, signal: dict[str, Any], playback: dict[str, Any]) -> ReactionEvent | None:
        now = self.clock()
        current_track_id = self._string_value(playback.get("current_track_id"))
        deferred = self._consume_deferred(current_track_id, playback)
        if deferred is not None:
            self._cooldown_until = now + self.cooldown_seconds
            self._fired_track_id = current_track_id
            return deferred
        if current_track_id != self._fired_track_id:
            self._fired_track_id = None
        if not current_track_id or playback.get("pending_queue_track_ids"):
            self._negative_since = None
            return None
        if not self._is_negative(signal):
            self._negative_since = None
            return None
        if self._negative_since is None:
            self._negative_since = now
            return None
        duration = now - self._negative_since
        if duration < self.negative_seconds or now < self._cooldown_until or self._fired_track_id == current_track_id:
            return None
        event = ReactionEvent(
            event_type="sustained_negative_reaction",
            current_track_id=current_track_id,
            current_cluster=self._string_value(playback.get("current_cluster")),
            duration_seconds=duration,
            signal=dict(signal),
        )
        if self._is_late(playback):
            self._deferred_event = event
            self._negative_since = None
            return None
        self._cooldown_until = now + self.cooldown_seconds
        self._fired_track_id = current_track_id
        return event

    def _consume_deferred(self, current_track_id: str | None, playback: dict[str, Any]) -> ReactionEvent | None:
        event = self._deferred_event
        if event is None or current_track_id is None:
            return None
        if current_track_id == event.current_track_id or playback.get("pending_queue_track_ids") or self._is_late(playback):
            return None
        self._deferred_event = None
        metadata = {**(event.metadata or {}), "deferred_from_track_id": event.current_track_id}
        return ReactionEvent(
            event_type=event.event_type,
            current_track_id=current_track_id,
            current_cluster=self._string_value(playback.get("current_cluster")) or event.current_cluster,
            duration_seconds=event.duration_seconds,
            signal=event.signal,
            metadata=metadata,
        )

    def _is_negative(self, signal: dict[str, Any]) -> bool:
        if signal.get("trend") != "negative":
            return False
        if self._float_value(signal.get("confidence")) < self.confidence_threshold:
            return False
        score = signal.get("score")
        if score is None:
            return True
        return self._float_value(score) <= self.negative_score_threshold

    def _is_late(self, playback: dict[str, Any]) -> bool:
        duration_ms = self._float_value(playback.get("duration_ms"))
        progress_ms = self._float_value(playback.get("progress_ms"))
        if duration_ms <= 0:
            return False
        return progress_ms / duration_ms >= self.defer_after_progress

    @staticmethod
    def _float_value(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _string_value(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        return text or None


class ClusterPolicyMonitor:
    def __init__(
        self,
        *,
        min_cluster_run: int | None = None,
        max_cluster_run: int = 6,
        defer_after_progress: float = 0.75,
        choose_target: Callable[[int, int], int] | None = None,
    ) -> None:
        self.min_cluster_run = max(1, int(min_cluster_run if min_cluster_run is not None else max_cluster_run))
        self.max_cluster_run = max(self.min_cluster_run, int(max_cluster_run))
        self.defer_after_progress = defer_after_progress
        self.choose_target = choose_target or random.randint
        self._target_cluster: str | None = None
        self._target_cluster_run: int | None = None
        self._fired_track_id: str | None = None
        self._deferred_event: ReactionEvent | None = None

    async def poll(self, playback: dict[str, Any]) -> ReactionEvent | None:
        current_track_id = ReactionMonitor._string_value(playback.get("current_track_id"))
        current_cluster = ReactionMonitor._string_value(playback.get("current_cluster"))
        deferred = self._consume_deferred(current_track_id, playback)
        if deferred is not None:
            self._fired_track_id = current_track_id
            return deferred
        if not current_track_id or playback.get("pending_queue_track_ids"):
            return None
        target_cluster_run = self._target_run(current_cluster or current_track_id)
        if current_track_id != self._fired_track_id:
            self._fired_track_id = None
        if current_track_id == self._fired_track_id:
            return None
        cluster_streak = int(ReactionMonitor._float_value(playback.get("cluster_streak")))
        if cluster_streak < target_cluster_run:
            return None
        event = ReactionEvent(
            event_type="max_cluster_streak_reached",
            current_track_id=current_track_id,
            current_cluster=current_cluster,
            duration_seconds=0.0,
            signal={"trend": "neutral", "confidence": 1.0, "score": 0.0, "source": "cluster_policy"},
            metadata={
                "cluster_streak": cluster_streak,
                "target_cluster_run": target_cluster_run,
                "min_cluster_run": self.min_cluster_run,
                "max_cluster_run": self.max_cluster_run,
            },
        )
        if self._is_late(playback):
            self._deferred_event = event
            return None
        self._fired_track_id = current_track_id
        return event

    def _consume_deferred(self, current_track_id: str | None, playback: dict[str, Any]) -> ReactionEvent | None:
        event = self._deferred_event
        if event is None or current_track_id is None:
            return None
        if current_track_id == event.current_track_id or playback.get("pending_queue_track_ids") or self._is_late(playback):
            return None
        self._deferred_event = None
        metadata = {**(event.metadata or {}), "deferred_from_track_id": event.current_track_id}
        return ReactionEvent(
            event_type=event.event_type,
            current_track_id=current_track_id,
            current_cluster=ReactionMonitor._string_value(playback.get("current_cluster")) or event.current_cluster,
            duration_seconds=event.duration_seconds,
            signal=event.signal,
            metadata=metadata,
        )

    def _is_late(self, playback: dict[str, Any]) -> bool:
        duration_ms = ReactionMonitor._float_value(playback.get("duration_ms"))
        progress_ms = ReactionMonitor._float_value(playback.get("progress_ms"))
        if duration_ms <= 0:
            return False
        return progress_ms / duration_ms >= self.defer_after_progress

    def _target_run(self, cluster: str) -> int:
        if cluster != self._target_cluster or self._target_cluster_run is None:
            self._target_cluster = cluster
            target = self.choose_target(self.min_cluster_run, self.max_cluster_run)
            self._target_cluster_run = max(self.min_cluster_run, min(self.max_cluster_run, int(target)))
        return self._target_cluster_run
