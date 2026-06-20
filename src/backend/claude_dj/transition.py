from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .observability import add_breadcrumb, capture_swallowed_exception, capture_warning, observe_async


@dataclass
class TransitionPlan:
    current_track_id: str
    next_track_id: str
    track_ids: list[str]
    narration_id: str
    ready: bool = True


class InMemoryTransitionStore:
    def __init__(self) -> None:
        self._plan: TransitionPlan | None = None

    def save(self, plan: TransitionPlan) -> None:
        add_breadcrumb(
            "Saved ready transition plan",
            category="claude_dj.transition",
            data={
                "current_track_id": plan.current_track_id,
                "next_track_id": plan.next_track_id,
                "track_count": len(plan.track_ids),
                "ready": plan.ready,
            },
        )
        self._plan = plan

    def get_ready_plan(self, ended_track_id: str) -> TransitionPlan | None:
        if self._plan is None:
            return None
        if self._plan.current_track_id != ended_track_id or not self._plan.ready:
            return None
        return self._plan

    def clear(self, ended_track_id: str) -> None:
        if self._plan and self._plan.current_track_id == ended_track_id:
            add_breadcrumb(
                "Cleared transition plan",
                category="claude_dj.transition",
                data={"ended_track_id": ended_track_id},
            )
            self._plan = None


class BoundaryAdapter(Protocol):
    async def get_music_volume(self) -> int: ...

    async def set_music_volume(self, volume_percent: int) -> None: ...

    async def play_track(self, track_id: str) -> None: ...

    async def play_narration(self, narration_id: str) -> None: ...


class BoundaryExecutor:
    def __init__(self, store: InMemoryTransitionStore, adapter: BoundaryAdapter) -> None:
        self.store = store
        self.adapter = adapter

    async def on_track_boundary(self, ended_track_id: str) -> None:
        async def run() -> None:
            plan = self.store.get_ready_plan(ended_track_id)
            if plan is None:
                add_breadcrumb(
                    "No ready transition plan at boundary; using deterministic fallback",
                    category="claude_dj.transition",
                    data={"ended_track_id": ended_track_id},
                    level="warning",
                )
                return

            original_volume = await self.adapter.get_music_volume()
            try:
                await self.adapter.set_music_volume(10)
                await self.adapter.play_track(plan.next_track_id)
                await self.adapter.play_narration(plan.narration_id)
            except Exception as exc:
                capture_swallowed_exception(
                    exc,
                    operation="claude_dj.transition.execute_ready_plan",
                    data={
                        "ended_track_id": ended_track_id,
                        "next_track_id": plan.next_track_id,
                        "narration_id": plan.narration_id,
                    },
                )
                raise
            finally:
                try:
                    await self.adapter.set_music_volume(original_volume)
                except Exception as exc:
                    capture_warning(
                        "Failed to restore music volume after boundary transition",
                        operation="claude_dj.transition.restore_volume",
                        data={"ended_track_id": ended_track_id, "original_volume": original_volume},
                    )
                    raise
            self.store.clear(ended_track_id)

        await observe_async(
            "claude_dj.transition.on_track_boundary",
            op="playback.boundary",
            data={"ended_track_id": ended_track_id},
            callback=run,
        )
