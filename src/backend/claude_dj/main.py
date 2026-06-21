from __future__ import annotations

import asyncio
import argparse
import os
import sys
from dataclasses import dataclass
from typing import TextIO

from dotenv import load_dotenv

from .observability import init_sentry

from .agent.client import ClaudeDJ
from .agent.runner import DJAgentRunner
from .mcp.narration import DeepgramNarrator, EphemeralNarrationStore, LocalNarrationPlayer
from .mcp.playback import InMemoryPlaybackRuntime
from .mcp.recommendations import RedisRecommendationClient
from .mcp.spotify import SpotifyConfig, SpotifyWebAPIPlayer
from .mascot import MascotAppProcess
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

    async def play_next_queued_track(self) -> str | None:
        result = await self.playback.play_next_queued_track()
        if result is None:
            return None
        print(f"play queued track: {result['track_id']}", file=self.output)
        return str(result["track_id"])

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


@dataclass(frozen=True)
class DJHarness:
    runner: DJAgentRunner
    playback: InMemoryPlaybackRuntime


class TrackBoundaryWatcher:
    def __init__(self) -> None:
        self._current_track_id: str | None = None
        self._boundary_handled = False
        self._last_seconds_remaining: int | None = None
        self._last_progress_ms: int | None = None
        self._last_was_playing = False

    async def maybe_handle_boundary(self, playback: InMemoryPlaybackRuntime, runner: DJAgentRunner) -> bool:
        state = await playback.get_current_playback()
        current_track_id = state.get("current_track_id")
        if current_track_id != self._current_track_id:
            self._current_track_id = current_track_id
            self._boundary_handled = False
            self._last_seconds_remaining = None
            self._last_progress_ms = None
            self._last_was_playing = False
        if not current_track_id or self._boundary_handled:
            self._remember_state(state)
            return False
        if not self._is_boundary_state(state):
            self._remember_state(state)
            return False

        await runner.on_track_boundary(ended_track_id=current_track_id)
        self._boundary_handled = True
        self._remember_state(state)
        return True

    def _is_boundary_state(self, state: dict[str, object]) -> bool:
        if self._int_value(state.get("seconds_remaining")) <= 0:
            return True
        return self._spotify_reset_after_near_end(state)

    def _spotify_reset_after_near_end(self, state: dict[str, object]) -> bool:
        queue_ready = bool(state.get("queue_track_ids") or state.get("pending_queue_track_ids"))
        if not queue_ready or bool(state.get("is_playing")):
            return False
        if self._int_value(state.get("progress_ms")) != 0:
            return False
        return self._last_was_playing

    def _remember_state(self, state: dict[str, object]) -> None:
        self._last_seconds_remaining = self._int_value(state.get("seconds_remaining"))
        self._last_progress_ms = self._int_value(state.get("progress_ms"))
        self._last_was_playing = bool(state.get("is_playing"))

    @staticmethod
    def _int_value(value: object) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0


def build_harness(
    *,
    output: TextIO = sys.stdout,
    verbose_claude: bool = False,
    demo_track_seconds: int | None = None,
) -> DJHarness:
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
    playback = InMemoryPlaybackRuntime(
        spotify=spotify,
        recommendations=RedisRecommendationClient(),
        initial_seed_track_id=os.environ.get("CLAUDE_DJ_INITIAL_REDIS_TRACK_ID", "deezer:100814018"),
        require_recommendations=env_flag("CLAUDE_DJ_REQUIRE_REDIS_RECOMMENDATIONS"),
        demo_track_seconds=demo_track_seconds,
    )
    narrator = DeepgramNarrator(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        model=os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-luna-en"),
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
    return DJHarness(DJAgentRunner(agent, boundary), playback)


def build_runner(
    *,
    output: TextIO = sys.stdout,
    verbose_claude: bool = False,
    demo_track_seconds: int | None = None,
) -> DJAgentRunner:
    return build_harness(
        output=output,
        verbose_claude=verbose_claude,
        demo_track_seconds=demo_track_seconds,
    ).runner


async def run_forever(
    *,
    output: TextIO = sys.stdout,
    sleep_seconds: float = 5.0,
    verbose_claude: bool = False,
    launch_mascot: bool = True,
    demo_track_seconds: int | None = None,
) -> None:
    harness: DJHarness | None = None
    mascot = MascotAppProcess() if launch_mascot else None
    boundary_watcher = TrackBoundaryWatcher()

    async def run_startup() -> DJHarness:
        harness = build_harness(
            output=output,
            verbose_claude=verbose_claude,
            demo_track_seconds=demo_track_seconds,
        )
        print("ClaudeDJ autonomous harness starting", file=output)
        await harness.runner.connect()
        await harness.runner.on_start()
        print("ClaudeDJ startup turn completed", file=output)
        return harness

    try:
        if mascot is not None:
            mascot.start()

        harness = await observe_async(
            "claude_dj.harness.startup",
            op="claude_dj.harness",
            data={"sleep_seconds": sleep_seconds},
            callback=run_startup,
        )

        while True:
            await asyncio.sleep(sleep_seconds)
            if await boundary_watcher.maybe_handle_boundary(harness.playback, harness.runner):
                print("ClaudeDJ track boundary handled", file=output)
                continue
            print("ClaudeDJ mid-song prepare starting", file=output)
            await harness.runner.on_mid_song_prepare(progress_percent=55)
            print("ClaudeDJ mid-song prepare completed", file=output)
    finally:
        if harness is not None:
            await harness.runner.disconnect()
        if mascot is not None:
            mascot.stop()


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return int(value)


def main(argv: list[str] | None = None) -> None:
    load_dotenv(".env")
    init_sentry()

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
    parser.add_argument(
        "--no-mascot",
        action="store_true",
        default=env_flag("CLAUDE_DJ_NO_MASCOT"),
        help="Run the harness without launching the desktop mascot app.",
    )
    parser.add_argument(
        "--demo-track-seconds",
        type=int,
        default=env_int("CLAUDE_DJ_DEMO_TRACK_SECONDS"),
        help="Cap each track's effective playback duration for demos, e.g. 30.",
    )
    args = parser.parse_args(argv)

    try:
        asyncio.run(
            run_forever(
                sleep_seconds=args.sleep_seconds,
                verbose_claude=args.verbose_claude,
                launch_mascot=not args.no_mascot,
                demo_track_seconds=args.demo_track_seconds,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        capture_swallowed_exception(exc, operation="claude_dj.cli.main")
        raise


if __name__ == "__main__":
    main()
