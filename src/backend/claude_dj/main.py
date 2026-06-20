from __future__ import annotations

import asyncio
import os
import sys
from typing import TextIO

from .observability import init_sentry

init_sentry()

from .agent.client import ClaudeDJ
from .agent.runner import DJAgentRunner
from .mcp.narration import DeepgramNarrator, EphemeralNarrationStore
from .observability import capture_swallowed_exception, capture_warning, observe_async
from .transition import BoundaryExecutor, InMemoryTransitionStore


class ConsoleBoundaryAdapter:
    def __init__(self, output: TextIO, narration_store: EphemeralNarrationStore) -> None:
        self.output = output
        self.narration_store = narration_store
        self.volume = 100

    async def get_music_volume(self) -> int:
        return self.volume

    async def set_music_volume(self, volume_percent: int) -> None:
        self.volume = volume_percent
        print(f"music volume: {volume_percent}%", file=self.output)

    async def play_track(self, track_id: str) -> None:
        print(f"play prepared track: {track_id}", file=self.output)

    async def play_narration(self, narration_id: str) -> None:
        if self.narration_store.get(narration_id) is None:
            capture_warning(
                "Prepared narration was missing at playback boundary",
                operation="claude_dj.playback.missing_narration",
                data={"narration_id": narration_id},
            )
        self.narration_store.delete(narration_id)
        print(f"play prepared narration: {narration_id}", file=self.output)


def build_runner(*, output: TextIO = sys.stdout) -> DJAgentRunner:
    transition_store = InMemoryTransitionStore()
    narration_store = EphemeralNarrationStore()
    narrator = DeepgramNarrator(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        model=os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-apollo-en"),
        speed=float(os.environ.get("DEEPGRAM_TTS_SPEED", "1.3")),
        store=narration_store,
    )
    agent = ClaudeDJ.create(transition_store, narrator)
    boundary = BoundaryExecutor(transition_store, ConsoleBoundaryAdapter(output, narration_store))
    return DJAgentRunner(agent, boundary)


async def run_forever(*, output: TextIO = sys.stdout, sleep_seconds: float = 5.0) -> None:
    async def run_startup() -> DJAgentRunner:
        runner = build_runner(output=output)
        print("ClaudeDJ autonomous harness starting", file=output)
        await runner.on_start()
        return runner

    runner = await observe_async(
        "claude_dj.harness.startup",
        op="claude_dj.harness",
        data={"sleep_seconds": sleep_seconds},
        callback=run_startup,
    )

    while True:
        await asyncio.sleep(sleep_seconds)
        await runner.on_mid_song_prepare(progress_percent=55)


def main() -> None:
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        capture_swallowed_exception(exc, operation="claude_dj.cli.main")
        raise


if __name__ == "__main__":
    main()
