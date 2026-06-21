from __future__ import annotations

import asyncio
import argparse
import os
import sys
from typing import TextIO

from .observability import init_sentry

init_sentry()

from .agent.client import ClaudeDJ
from .agent.runner import DJAgentRunner
from .mcp.narration import DeepgramNarrator, EphemeralNarrationStore, LocalNarrationPlayer
from .mcp.playback import InMemoryPlaybackRuntime
from .mcp.spotify import SpotifyConfig, SpotifyWebAPIPlayer
from .observability import capture_swallowed_exception, capture_warning, observe_async
from .transition import BoundaryExecutor, InMemoryTransitionStore


class ConsoleBoundaryAdapter:
    def __init__(
        self,
        output: TextIO,
        narration_store: EphemeralNarrationStore,
        narration_player: LocalNarrationPlayer,
        playback: InMemoryPlaybackRuntime,
    ) -> None:
        self.output = output
        self.narration_store = narration_store
        self.narration_player = narration_player
        self.playback = playback
        self.volume = 100

    async def get_music_volume(self) -> int:
        return self.volume

    async def set_music_volume(self, volume_percent: int) -> None:
        self.volume = volume_percent
        print(f"music volume: {volume_percent}%", file=self.output)

    async def play_track(self, track_id: str) -> None:
        await self.playback.play_track(track_id)
        print(f"play prepared track: {track_id}", file=self.output)

    async def play_narration(self, narration_id: str) -> None:
        narration = self.narration_store.get(narration_id)
        if narration is None:
            capture_warning(
                "Prepared narration was missing at playback boundary",
                operation="claude_dj.playback.missing_narration",
                data={"narration_id": narration_id},
            )
        else:
            self.narration_player.play(narration)
        self.narration_store.delete(narration_id)
        print(f"play prepared narration: {narration_id}", file=self.output)


def build_runner(*, output: TextIO = sys.stdout, verbose_claude: bool = False) -> DJAgentRunner:
    transition_store = InMemoryTransitionStore()
    narration_store = EphemeralNarrationStore()
    narration_player = LocalNarrationPlayer()
    spotify = SpotifyWebAPIPlayer(
        SpotifyConfig(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
            refresh_token=os.environ["SPOTIFY_REFRESH_TOKEN"],
        )
    )
    playback = InMemoryPlaybackRuntime(spotify=spotify)
    narrator = DeepgramNarrator(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        model=os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-apollo-en"),
        speed=float(os.environ.get("DEEPGRAM_TTS_SPEED", "1.3")),
        store=narration_store,
    )
    agent = ClaudeDJ.create(
        transition_store,
        narrator,
        playback,
        narration_player,
        output=output,
        verbose_claude=verbose_claude,
    )
    boundary = BoundaryExecutor(
        transition_store,
        ConsoleBoundaryAdapter(output, narration_store, narration_player, playback),
    )
    return DJAgentRunner(agent, boundary)


async def run_forever(
    *,
    output: TextIO = sys.stdout,
    sleep_seconds: float = 5.0,
    verbose_claude: bool = False,
) -> None:
    async def run_startup() -> DJAgentRunner:
        runner = build_runner(output=output, verbose_claude=verbose_claude)
        print("ClaudeDJ autonomous harness starting", file=output)
        await runner.connect()
        await runner.on_start()
        print("ClaudeDJ startup turn completed", file=output)
        return runner

    runner = await observe_async(
        "claude_dj.harness.startup",
        op="claude_dj.harness",
        data={"sleep_seconds": sleep_seconds},
        callback=run_startup,
    )

    try:
        while True:
            await asyncio.sleep(sleep_seconds)
            print("ClaudeDJ mid-song prepare starting", file=output)
            await runner.on_mid_song_prepare(progress_percent=55)
            print("ClaudeDJ mid-song prepare completed", file=output)
    finally:
        await runner.disconnect()


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the long-running ClaudeDJ harness.")
    parser.add_argument(
        "--verbose-claude",
        action="store_true",
        default=env_flag("CLAUDE_DJ_VERBOSE_CLAUDE"),
        help="Print every Claude SDK message, tool call, tool result, and rate-limit event.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=float(os.environ.get("CLAUDE_DJ_LOOP_SLEEP_SECONDS", "5")),
        help="Seconds between mid-song prepare turns.",
    )
    args = parser.parse_args(argv)

    try:
        asyncio.run(run_forever(sleep_seconds=args.sleep_seconds, verbose_claude=args.verbose_claude))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        capture_swallowed_exception(exc, operation="claude_dj.cli.main")
        raise


if __name__ == "__main__":
    main()
