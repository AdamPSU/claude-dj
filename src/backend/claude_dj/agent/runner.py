from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from ..observability import observe_async, observe_run


class Agent(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def handle_start(self) -> None: ...

    async def handle_reaction_event(self, event: dict[str, object]) -> None: ...

    async def handle_queue_refresh(self, playback: dict[str, object]) -> None: ...


class Boundary(Protocol):
    async def on_track_boundary(self, ended_track_id: str) -> None: ...


class DJAgentRunner:
    def __init__(self, agent: Agent, boundary: Boundary) -> None:
        self.agent = agent
        self.boundary = boundary
        self.session_id = uuid4().hex

    async def connect(self) -> None:
        await self.agent.connect()

    async def disconnect(self) -> None:
        await self.agent.disconnect()

    async def on_start(self) -> None:
        await observe_run(
            "on_start",
            session_id=self.session_id,
            data={"hook": "on_start"},
            callback=self.agent.handle_start,
        )

    async def on_reaction_event(self, event: dict[str, object]) -> None:
        await observe_run(
            "on_reaction_event",
            session_id=self.session_id,
            data={"hook": "on_reaction_event", "event_type": str(event.get("event_type", "reaction_event"))},
            callback=lambda: self.agent.handle_reaction_event(event),
        )

    async def on_queue_refresh(self, playback: dict[str, object]) -> None:
        await observe_run(
            "on_queue_refresh",
            session_id=self.session_id,
            data={"hook": "on_queue_refresh", "current_track_id": str(playback.get("current_track_id", ""))},
            callback=lambda: self.agent.handle_queue_refresh(playback),
        )

    async def on_track_boundary(self, *, ended_track_id: str) -> None:
        await observe_async(
            "claude_dj.runner.on_track_boundary",
            op="claude_dj.runner",
            data={"ended_track_id": ended_track_id},
            callback=lambda: self.boundary.on_track_boundary(ended_track_id),
        )
