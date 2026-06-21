from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

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

    async def play_next_queued_track(self) -> str | None: ...

    async def pause_music(self) -> None: ...

    async def resume_music(self) -> None: ...

    async def play_narration(self, narration_id: str) -> None: ...


class BoundaryExecutor:
    def __init__(
        self,
        store: InMemoryTransitionStore,
        adapter: BoundaryAdapter,
        *,
        fade_seconds: float = 1.0,
        fade_steps: int = 10,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.fade_seconds = max(0.0, fade_seconds)
        self.fade_steps = max(0, fade_steps)
        self.sleep = sleep

    async def on_track_boundary(self, ended_track_id: str) -> None:
        async def run() -> None:
            plan = self.store.get_ready_plan(ended_track_id)
            if plan is None:
                next_track_id = await self._play_next_with_fade()
                add_breadcrumb(
                    "No ready transition plan at boundary; using deterministic fallback",
                    category="claude_dj.transition",
                    data={"ended_track_id": ended_track_id, "next_track_id": next_track_id},
                    level="info" if next_track_id else "warning",
                )
                return

            original_volume = await self._fade_out()
            try:
                await self.adapter.play_track(plan.next_track_id)
                await self.adapter.pause_music()
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
                    await self.adapter.resume_music()
                except Exception as exc:
                    capture_warning(
                        "Failed to resume music after boundary transition",
                        operation="claude_dj.transition.resume_music",
                        data={"ended_track_id": ended_track_id},
                    )
                    raise
                finally:
                    await self._fade_in(original_volume)
            self.store.clear(ended_track_id)

        await observe_async(
            "claude_dj.transition.on_track_boundary",
            op="playback.boundary",
            data={"ended_track_id": ended_track_id},
            callback=run,
        )

    async def _play_next_with_fade(self) -> str | None:
        original_volume = await self._fade_out()
        try:
            return await self.adapter.play_next_queued_track()
        finally:
            await self._fade_in(original_volume)

    async def _fade_out(self) -> int:
        original_volume = self._volume_percent(await self.adapter.get_music_volume())
        await self._fade_music(original_volume, 0)
        return original_volume

    async def _fade_in(self, volume_percent: int) -> None:
        await self._fade_music(0, self._volume_percent(volume_percent))

    async def _fade_music(self, start_volume: int, end_volume: int) -> None:
        if start_volume == end_volume:
            return
        if self.fade_seconds <= 0 or self.fade_steps <= 0:
            await self.adapter.set_music_volume(end_volume)
            return
        delay = self.fade_seconds / self.fade_steps
        for step in range(1, self.fade_steps + 1):
            volume = round(start_volume + ((end_volume - start_volume) * step / self.fade_steps))
            await self.adapter.set_music_volume(self._volume_percent(volume))
            await self.sleep(delay)

    @staticmethod
    def _volume_percent(value: int) -> int:
        return max(0, min(100, int(value)))
