from __future__ import annotations

import asyncio
import argparse
import os
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import TextIO

from dotenv import load_dotenv

from .observability import init_sentry

from .agent.client import ClaudeDJ
from .agent.runner import DJAgentRunner
from .mcp.handlers import ReactionSource
from .mcp.narration import DeepgramNarrator, EphemeralNarrationStore, LocalNarrationPlayer, NarrationPlayer
from .mcp.playback import InMemoryPlaybackRuntime
from .mcp.recommendations import RedisRecommendationClient
from .mcp.spotify import SpotifyConfig, SpotifyWebAPIPlayer
from .mascot import MascotAppProcess, MascotNarrationPlayer
from .observability import capture_swallowed_exception, capture_warning, observe_async
from .reactions.monitor import ClusterPolicyMonitor, ReactionMonitor
from .reactions.reactor import Reactor, ReactorReactionSource
from .reactions.webcam import DEFAULT_FACE_MODEL_PATH, WebcamWorker
from .transition import BoundaryExecutor, InMemoryTransitionStore

DEFAULT_QUEUE_MIN_TRACKS = 2
DEFAULT_QUEUE_MAX_TRACKS = 4
DEFAULT_MIN_CLUSTER_RUN = 2
DEFAULT_MAX_CLUSTER_RUN = 4


class ConsoleBoundaryAdapter:
    def __init__(
        self,
        output: TextIO,
        narration_store: EphemeralNarrationStore,
        narration_player: NarrationPlayer,
        playback: InMemoryPlaybackRuntime,
    ) -> None:
        self.output = output
        self.narration_store = narration_store
        self.narration_player = narration_player
        self.playback = playback

    async def get_music_volume(self) -> int:
        return await self.playback.get_music_volume()

    async def set_music_volume(self, volume_percent: int) -> None:
        await self.playback.set_music_volume(volume_percent)

    async def pause_music(self) -> None:
        await self.playback.pause_music()
        print("pause music for bridge narration", file=self.output)

    async def resume_music(self) -> None:
        await self.playback.resume_music()
        print("resume music after bridge narration", file=self.output)

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
    reaction_source: ReactionSource
    reaction_runtime: ReactionRuntime | None = None


@dataclass(frozen=True)
class ReactionRuntime:
    source: ReactionSource
    reactor: Reactor | None = None
    preview_worker: WebcamWorker | None = None

    def start(self) -> None:
        if self.reactor is not None:
            self.reactor.start()

    def stop(self) -> None:
        if self.reactor is not None:
            self.reactor.stop()

    def pump_preview(self) -> bool:
        return bool(self.preview_worker and self.preview_worker.pump_preview_window())


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
            return self._has_next_track(state)
        return self._spotify_reset_after_near_end(state)

    def _has_next_track(self, state: dict[str, object]) -> bool:
        return bool(state.get("queue_track_ids") or state.get("pending_queue_track_ids"))

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


class QueueRefreshMonitor:
    def __init__(self) -> None:
        self._refreshed_track_id: str | None = None

    def should_refresh(self, playback: dict[str, object]) -> bool:
        current_track_id = playback.get("current_track_id")
        if not queue_needs_refresh(playback):
            if current_track_id != self._refreshed_track_id:
                self._refreshed_track_id = None
            return False
        if current_track_id == self._refreshed_track_id:
            return False
        self._refreshed_track_id = str(current_track_id)
        return True


def build_harness(
    *,
    output: TextIO = sys.stdout,
    verbose_claude: bool = False,
    demo_track_seconds: int | None = None,
    mascot: MascotAppProcess | None = None,
) -> DJHarness:
    transition_store = InMemoryTransitionStore()
    narration_store = EphemeralNarrationStore()
    local_narration_player = LocalNarrationPlayer()
    narration_player: NarrationPlayer = (
        MascotNarrationPlayer(local_narration_player, mascot) if mascot is not None else local_narration_player
    )
    spotify = SpotifyWebAPIPlayer(
        SpotifyConfig(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
            refresh_token=os.environ["SPOTIFY_REFRESH_TOKEN"],
        )
    )
    recommendations = RedisRecommendationClient()
    playback = InMemoryPlaybackRuntime(
        spotify=spotify,
        recommendations=recommendations,
        initial_seed_track_id=resolve_initial_seed_track_id(recommendations),
        require_recommendations=env_flag("CLAUDE_DJ_REQUIRE_REDIS_RECOMMENDATIONS"),
        demo_track_seconds=demo_track_seconds,
        queue_min_tracks=env_int("CLAUDE_DJ_QUEUE_MIN_TRACKS") or DEFAULT_QUEUE_MIN_TRACKS,
        queue_max_tracks=env_int("CLAUDE_DJ_QUEUE_MAX_TRACKS") or DEFAULT_QUEUE_MAX_TRACKS,
    )
    narrator = DeepgramNarrator(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        model=os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-luna-en"),
        speed=float(os.environ.get("DEEPGRAM_TTS_SPEED", "1.3")),
        store=narration_store,
    )
    reaction_runtime = build_reaction_runtime()
    agent = ClaudeDJ.create(
        transition_store,
        narrator,
        playback,
        narration_player,
        reaction_runtime.source,
        output=output,
        verbose_claude=verbose_claude,
    )
    boundary = BoundaryExecutor(
        transition_store,
        ConsoleBoundaryAdapter(output, narration_store, narration_player, playback),
    )
    return DJHarness(DJAgentRunner(agent, boundary), playback, reaction_runtime.source, reaction_runtime)


def build_reaction_runtime() -> ReactionRuntime:
    frame_source: WebcamWorker | None = None
    if not env_flag("CLAUDE_DJ_NO_WEBCAM"):
        frame_source = WebcamWorker(
            camera_index=env_int("CLAUDE_DJ_CAMERA_INDEX") or 0,
            model_path=os.environ.get("CLAUDE_DJ_FACE_LANDMARKER_MODEL", str(DEFAULT_FACE_MODEL_PATH)),
            show_preview=True,
        )
    reactor = Reactor(frame_source=frame_source)
    return ReactionRuntime(source=ReactorReactionSource(reactor), reactor=reactor, preview_worker=frame_source)


def build_runner(
    *,
    output: TextIO = sys.stdout,
    verbose_claude: bool = False,
    demo_track_seconds: int | None = None,
    mascot: MascotAppProcess | None = None,
) -> DJAgentRunner:
    return build_harness(
        output=output,
        verbose_claude=verbose_claude,
        demo_track_seconds=demo_track_seconds,
        mascot=mascot,
    ).runner


async def run_forever(
    *,
    output: TextIO = sys.stdout,
    sleep_seconds: float = 1.0,
    verbose_claude: bool = False,
    launch_mascot: bool = True,
    demo_track_seconds: int | None = None,
) -> None:
    harness: DJHarness | None = None
    mascot = MascotAppProcess() if launch_mascot else None
    reaction_preview_task: asyncio.Task[None] | None = None
    boundary_watcher = TrackBoundaryWatcher()
    queue_refresh_monitor = QueueRefreshMonitor()
    reaction_monitor = ReactionMonitor(
        negative_seconds=env_float("CLAUDE_DJ_NEGATIVE_REACTION_SECONDS", 5.0),
        confidence_threshold=env_float("CLAUDE_DJ_NEGATIVE_CONFIDENCE_THRESHOLD", 0.6),
        negative_score_threshold=env_float("CLAUDE_DJ_NEGATIVE_SCORE_THRESHOLD", 0.4),
        cooldown_seconds=env_float("CLAUDE_DJ_REACTION_SHIFT_COOLDOWN_SECONDS", 45.0),
    )
    cluster_policy_monitor = build_cluster_policy_monitor()

    async def run_startup() -> DJHarness:
        nonlocal harness, reaction_preview_task
        harness = build_harness(
            output=output,
            verbose_claude=verbose_claude,
            demo_track_seconds=demo_track_seconds,
            mascot=mascot,
        )
        print("ClaudeDJ autonomous harness starting", file=output)
        if harness.reaction_runtime is not None:
            harness.reaction_runtime.start()
            reaction_preview_task = asyncio.create_task(pump_reaction_preview(harness.reaction_runtime))
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
            await run_harness_tick(
                harness,
                boundary_watcher,
                reaction_monitor,
                output,
                queue_refresh_monitor,
                cluster_policy_monitor=cluster_policy_monitor,
            )
    finally:
        if reaction_preview_task is not None:
            reaction_preview_task.cancel()
            with suppress(asyncio.CancelledError):
                await reaction_preview_task
        if harness is not None:
            if harness.reaction_runtime is not None:
                harness.reaction_runtime.stop()
            await harness.runner.disconnect()
        if mascot is not None:
            mascot.stop()


# Default starting track ("Don't" by Bryson Tiller). Must stay in sync with
# recommendation_engine.config.DEFAULT_SEED_TRACK_ID.
DEFAULT_INITIAL_SEED_TRACK_ID = "deezer:100814018"


def resolve_initial_seed_track_id(recommendations: RedisRecommendationClient) -> str:
    """Resolve env override > imported session-history seed > default seed."""
    override = os.environ.get("CLAUDE_DJ_INITIAL_REDIS_TRACK_ID", "").strip()
    if override:
        return override
    imported = recommendations.get_initial_seed_track_id()
    if imported:
        return imported
    return DEFAULT_INITIAL_SEED_TRACK_ID


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


async def pump_reaction_preview(reaction_runtime: ReactionRuntime, sleep_seconds: float = 1.0 / 30.0) -> None:
    while True:
        reaction_runtime.pump_preview()
        await asyncio.sleep(sleep_seconds)


def env_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return float(value)


def build_cluster_policy_monitor() -> ClusterPolicyMonitor:
    return ClusterPolicyMonitor(
        min_cluster_run=env_int("CLAUDE_DJ_MIN_CLUSTER_RUN") or DEFAULT_MIN_CLUSTER_RUN,
        max_cluster_run=env_int("CLAUDE_DJ_MAX_CLUSTER_RUN") or DEFAULT_MAX_CLUSTER_RUN,
    )


async def run_harness_tick(
    harness: DJHarness,
    boundary_watcher: TrackBoundaryWatcher,
    reaction_monitor: ReactionMonitor,
    output: TextIO,
    queue_refresh_monitor: QueueRefreshMonitor | None = None,
    cluster_policy_monitor: ClusterPolicyMonitor | None = None,
) -> str:
    if await boundary_watcher.maybe_handle_boundary(harness.playback, harness.runner):
        print("ClaudeDJ track boundary handled", file=output)
        playback = await harness.playback.get_current_playback()
        log_queue_state("after_boundary", playback, output)
        if cluster_policy_monitor is not None:
            event = await cluster_policy_monitor.poll(playback)
            if event is not None:
                print("ClaudeDJ reaction event planning starting", file=output)
                await harness.runner.on_reaction_event(event.to_prompt_data())
                print("ClaudeDJ reaction event planning completed", file=output)
                log_queue_state("after_reaction_event", await harness.playback.get_current_playback(), output)
                return "boundary_reaction_event"
        queue_monitor = queue_refresh_monitor or QueueRefreshMonitor()
        if queue_monitor.should_refresh(playback):
            print("ClaudeDJ queue refresh planning starting", file=output)
            await harness.runner.on_queue_refresh(playback)
            print("ClaudeDJ queue refresh planning completed", file=output)
            log_queue_state("after_queue_refresh", await harness.playback.get_current_playback(), output)
            return "boundary_queue_refresh"
        return "boundary"
    playback = await harness.playback.get_current_playback()
    log_queue_state("tick", playback, output)
    signal = await harness.reaction_source.get_reaction_signal()
    event = await reaction_monitor.poll(signal, playback)
    if event is None and cluster_policy_monitor is not None:
        event = await cluster_policy_monitor.poll(playback)
    if event is None:
        queue_monitor = queue_refresh_monitor or QueueRefreshMonitor()
        if queue_monitor.should_refresh(playback):
            print("ClaudeDJ queue refresh planning starting", file=output)
            await harness.runner.on_queue_refresh(playback)
            print("ClaudeDJ queue refresh planning completed", file=output)
            log_queue_state("after_queue_refresh", await harness.playback.get_current_playback(), output)
            return "queue_refresh"
        return "idle"
    print("ClaudeDJ reaction event planning starting", file=output)
    await harness.runner.on_reaction_event(event.to_prompt_data())
    print("ClaudeDJ reaction event planning completed", file=output)
    return "reaction_event"


def queue_needs_refresh(playback: dict[str, object]) -> bool:
    return bool(playback.get("current_track_id")) and not bool(playback.get("queue_track_ids")) and not bool(
        playback.get("pending_queue_track_ids")
    )


def log_queue_state(label: str, playback: dict[str, object], output: TextIO) -> None:
    print(
        "ClaudeDJ queue state "
        f"{label} current={playback.get('current_track_id')} "
        f"queue={playback.get('queue_track_ids') or []} "
        f"pending={playback.get('pending_queue_track_ids') or []} "
        f"seconds_remaining={playback.get('seconds_remaining')}",
        file=output,
        flush=True,
    )


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
        default=float(os.environ.get("CLAUDE_DJ_LOOP_SLEEP_SECONDS", "1")),
        help="Seconds between boundary/reaction checks.",
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
        default=env_int("CLAUDE_DJ_DEMO_TRACK_SECONDS") or 20,
        help="Cap each track's effective playback duration for demos.",
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
