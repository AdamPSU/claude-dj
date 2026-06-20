from __future__ import annotations

from typing import Protocol

from ..observability import observe_async


class Agent(Protocol):
    async def handle_start(self) -> None: ...

    async def handle_mid_song_prepare(self, progress_percent: int) -> None: ...


class Boundary(Protocol):
    async def on_track_boundary(self, ended_track_id: str) -> None: ...


class DJAgentRunner:
    def __init__(self, agent: Agent, boundary: Boundary) -> None:
        self.agent = agent
        self.boundary = boundary

    async def on_start(self) -> None:
        await observe_async(
            "claude_dj.runner.on_start",
            op="claude_dj.runner",
            data={"hook": "on_start"},
            callback=self.agent.handle_start,
        )

    async def on_mid_song_prepare(self, *, progress_percent: int) -> None:
        await observe_async(
            "claude_dj.runner.on_mid_song_prepare",
            op="claude_dj.runner",
            data={"hook": "on_mid_song_prepare", "progress_percent": progress_percent},
            callback=lambda: self.agent.handle_mid_song_prepare(progress_percent=progress_percent),
        )

    async def on_track_boundary(self, *, ended_track_id: str) -> None:
        await observe_async(
            "claude_dj.runner.on_track_boundary",
            op="claude_dj.runner",
            data={"ended_track_id": ended_track_id},
            callback=lambda: self.boundary.on_track_boundary(ended_track_id),
        )
