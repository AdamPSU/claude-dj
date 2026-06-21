import asyncio
import io
import os
import time
import unittest
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from claude_dj.agent.client import ClaudeDJ
from claude_dj.agent.runner import DJAgentRunner
from claude_dj.main import TrackBoundaryWatcher
from claude_dj.mcp.narration import (
    DeepgramNarrator,
    DeepgramRequester,
    DeepgramResponse,
    EphemeralNarrationStore,
    NarrationAudio,
    UrlLibDeepgramRequester,
)
from claude_dj.mcp.playback import InMemoryPlaybackRuntime, SpotifyDevice, SpotifyPlaybackState, SpotifyPlaylist, Track
from claude_dj.mcp.recommendations import RedisRecommendationClient
from claude_dj.mcp.spotify import SpotifyConfig, SpotifyWebAPIPlayer
from claude_dj.transition import BoundaryExecutor, InMemoryTransitionStore


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


def live_e2e_enabled() -> bool:
    return os.environ.get("CLAUDE_DJ_LIVE_E2E", "").strip().lower() in {"1", "true", "yes", "on"}


def require_env(names: list[str]) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise unittest.SkipTest(f"missing live E2E environment variables: {', '.join(missing)}")


class FakeCameraReactionSource:
    def __init__(self) -> None:
        self.changed = False
        self.calls: list[dict[str, Any]] = []

    def radically_change(self) -> None:
        self.changed = True

    async def get_reaction_signal(self) -> dict[str, Any]:
        if self.changed:
            signal = {
                "available": True,
                "stub": False,
                "source": "fake_camera",
                "trend": "negative",
                "confidence": 0.99,
                "score": -0.95,
                "camera_state": "radical_change",
                "summary": "The fake camera feed changed abruptly and strongly rejected the current direction.",
            }
        else:
            signal = {
                "available": True,
                "stub": False,
                "source": "fake_camera",
                "trend": "positive",
                "confidence": 0.72,
                "score": 0.58,
                "camera_state": "settled",
                "summary": "The fake camera feed is calm and engaged.",
            }
        self.calls.append(signal)
        return dict(signal)


class RecordingDeepgramRequester:
    def __init__(self, inner: DeepgramRequester | None = None) -> None:
        self.inner = inner or UrlLibDeepgramRequester()
        self.texts: list[str] = []

    async def post(self, url: str, *, headers: dict[str, str], json: dict[str, str]) -> DeepgramResponse:
        self.texts.append(json["text"])
        return await self.inner.post(url, headers=headers, json=json)


class CapturingNarrationPlayer:
    def __init__(self) -> None:
        self.played: list[NarrationAudio] = []

    def play(self, narration: NarrationAudio) -> None:
        self.played.append(narration)


class VirtualDurationSpotifyPlayer:
    def __init__(self, inner: SpotifyWebAPIPlayer, *, duration_seconds: float = 15.0) -> None:
        self.inner = inner
        self.duration_ms = int(duration_seconds * 1000)
        self.started_uris: list[str] = []
        self.current_uri: str | None = None
        self.current_started_at = 0.0

    async def start_track(self, spotify_uri: str) -> None:
        await self.inner.start_track(spotify_uri)
        self.started_uris.append(spotify_uri)
        self.current_uri = spotify_uri
        self.current_started_at = time.monotonic()

    async def get_current_playback(self) -> SpotifyPlaybackState | None:
        real_state = await self.inner.get_current_playback()
        if self.current_uri is None:
            return real_state
        progress_ms = min(self.duration_ms, int((time.monotonic() - self.current_started_at) * 1000))
        return SpotifyPlaybackState(
            track_id=self.current_uri.rsplit(":", 1)[-1],
            spotify_uri=self.current_uri,
            progress_ms=progress_ms,
            duration_ms=self.duration_ms,
            is_playing=progress_ms < self.duration_ms,
            device=real_state.device if real_state else None,
        )

    async def search_tracks(self, query: str, limit: int = 6) -> list[Track]:
        return await self.inner.search_tracks(query, limit=limit)

    async def list_user_playlists(self, limit: int = 20) -> list[SpotifyPlaylist]:
        return await self.inner.list_user_playlists(limit=limit)

    async def list_playlist_tracks(self, playlist_id: str, playlist_name: str, limit: int = 100) -> list[Track]:
        return await self.inner.list_playlist_tracks(playlist_id, playlist_name, limit=limit)

    async def list_devices(self) -> list[SpotifyDevice]:
        return await self.inner.list_devices()

    async def transfer_playback(self, device_id: str, *, play: bool = False) -> None:
        await self.inner.transfer_playback(device_id, play=play)


class RecordingBoundaryAdapter:
    def __init__(
        self,
        playback: InMemoryPlaybackRuntime,
        narration_store: EphemeralNarrationStore,
        narration_player: CapturingNarrationPlayer,
    ) -> None:
        self.playback = playback
        self.narration_store = narration_store
        self.narration_player = narration_player
        self.volume = 100
        self.volume_events: list[int] = []
        self.prepared_tracks: list[str] = []
        self.fallback_tracks: list[str] = []
        self.boundary_narrations: list[str] = []

    async def get_music_volume(self) -> int:
        return self.volume

    async def set_music_volume(self, volume_percent: int) -> None:
        self.volume = volume_percent
        self.volume_events.append(volume_percent)

    async def play_track(self, track_id: str) -> None:
        await self.playback.play_track(track_id)
        self.prepared_tracks.append(track_id)

    async def play_next_queued_track(self) -> str | None:
        result = await self.playback.play_next_queued_track()
        if result is None:
            return None
        track_id = str(result["track_id"])
        self.fallback_tracks.append(track_id)
        return track_id

    async def play_narration(self, narration_id: str) -> None:
        narration = self.narration_store.get(narration_id)
        if narration is not None:
            self.narration_player.play(narration)
        self.narration_store.delete(narration_id)
        self.boundary_narrations.append(narration_id)


async def wait_for_boundary(
    watcher: TrackBoundaryWatcher,
    playback: InMemoryPlaybackRuntime,
    runner: DJAgentRunner,
    *,
    timeout_seconds: float = 25.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if await watcher.maybe_handle_boundary(playback, runner):
            return
        await asyncio.sleep(0.5)
    raise AssertionError("track boundary did not fire before timeout")


@unittest.skipUnless(live_e2e_enabled(), "set CLAUDE_DJ_LIVE_E2E=1 to run live agentic E2E")
class LiveAgenticPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        require_env(
            [
                "SPOTIFY_CLIENT_ID",
                "SPOTIFY_CLIENT_SECRET",
                "SPOTIFY_REFRESH_TOKEN",
                "DEEPGRAM_API_KEY",
                "REDIS_HOST",
                "REDIS_PORT",
                "REDIS_PASSWORD",
            ]
        )

    async def test_real_claude_session_responds_to_fake_camera_and_plays_three_fast_tracks(self) -> None:
        output = io.StringIO()
        transition_store = InMemoryTransitionStore()
        narration_store = EphemeralNarrationStore()
        narration_player = CapturingNarrationPlayer()
        camera = FakeCameraReactionSource()
        deepgram_requester = RecordingDeepgramRequester()
        spotify = VirtualDurationSpotifyPlayer(
            SpotifyWebAPIPlayer(
                SpotifyConfig(
                    client_id=os.environ["SPOTIFY_CLIENT_ID"],
                    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
                    refresh_token=os.environ["SPOTIFY_REFRESH_TOKEN"],
                    request_timeout_seconds=float(os.environ.get("SPOTIFY_REQUEST_TIMEOUT_SECONDS", "10")),
                )
            ),
            duration_seconds=15.0,
        )
        playback = InMemoryPlaybackRuntime(
            tracks=[],
            spotify=spotify,
            recommendations=RedisRecommendationClient(),
            initial_seed_track_id=os.environ.get("CLAUDE_DJ_INITIAL_REDIS_TRACK_ID", "deezer:100814018"),
            require_recommendations=True,
        )
        narrator = DeepgramNarrator(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            model=os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-luna-en"),
            speed=float(os.environ.get("DEEPGRAM_TTS_SPEED", "1.3")),
            store=narration_store,
            requester=deepgram_requester,
        )
        agent = ClaudeDJ.create(
            transition_store,
            narrator,
            playback,
            narration_player,
            camera,
            output=output,
            verbose_claude=True,
        )
        boundary_adapter = RecordingBoundaryAdapter(playback, narration_store, narration_player)
        boundary = BoundaryExecutor(transition_store, boundary_adapter)
        runner = DJAgentRunner(agent, boundary)
        watcher = TrackBoundaryWatcher()

        await agent.connect()
        try:
            await agent._send_turn(
                """
                Live E2E startup. Do not ask for user input.
                Use MCP tools only.
                Call get_session_context.
                Call search_track_embeddings with mode="similar", signal="positive", limit=3, exclude_recent=true.
                Choose exactly 3 playable candidates from that result.
                Call replace_queue with exactly those 3 ids and reason="live_e2e_start".
                Call narrate with mode="immediate" and reason="live_e2e_start".
                Call play_track for the first queued id.
                Stop after play_track.
                """
            )

            startup_playback = await playback.get_current_playback()
            self.assertTrue(startup_playback["current_track_id"])
            self.assertEqual(1 + len(startup_playback["queue_track_ids"]), 3)
            self.assertEqual(len(spotify.started_uris), 1)
            self.assertGreaterEqual(len(deepgram_requester.texts), 1)
            self.assertEqual(len(narration_player.played), 1)

            camera.radically_change()
            current_track_id = startup_playback["current_track_id"]
            current_cluster = startup_playback["current_cluster"]
            await agent._send_turn(
                f"""
                Live E2E mid-song preparation. The fake camera has radically changed.
                Use MCP tools only and do not interrupt the current song.
                Call get_current_playback, get_session_context, and get_reaction_signal.
                The expected reaction source is fake_camera with trend negative.
                Because the reaction is negative, call search_track_embeddings with seed_track_id="{current_track_id}",
                mode="shift", signal="negative", avoid_clusters=["{current_cluster}"], exclude_recent=true, limit=3.
                Choose exactly 3 shifted candidates.
                Call replace_queue with those ids, reason="live_e2e_camera_shift", timing="after_current_track".
                Call narrate with mode="prepare", reason="live_e2e_camera_shift", timing="after_current_track",
                current_track_id="{current_track_id}", next_track_id set to the first shifted candidate, and track_ids set to all shifted ids.
                Stop after the prepared narration is ready. Do not call play_track.
                """
            )

            self.assertGreaterEqual(len(camera.calls), 1)
            self.assertEqual(camera.calls[-1]["trend"], "negative")
            self.assertGreaterEqual(len(deepgram_requester.texts), 2)
            prepared_plan = transition_store.get_ready_plan(str(current_track_id))
            self.assertIsNotNone(prepared_plan)
            assert prepared_plan is not None
            self.assertEqual(len(prepared_plan.track_ids), 3)
            prepared_track_ids = list(prepared_plan.track_ids)
            self.assertEqual(playback.pending_queue_track_ids, prepared_track_ids)

            await wait_for_boundary(watcher, playback, runner)
            self.assertEqual(boundary_adapter.prepared_tracks, [prepared_track_ids[0]])
            self.assertEqual(boundary_adapter.boundary_narrations, [prepared_plan.narration_id])
            self.assertIn(10, boundary_adapter.volume_events)
            self.assertIn(100, boundary_adapter.volume_events)
            self.assertEqual(len(narration_player.played), 2)

            await wait_for_boundary(watcher, playback, runner)
            await wait_for_boundary(watcher, playback, runner)

            self.assertEqual(boundary_adapter.fallback_tracks, prepared_track_ids[1:])
            self.assertGreaterEqual(len(spotify.started_uris), 4)
            self.assertEqual(playback.queue_track_ids, [])
            self.assertEqual(playback.pending_queue_track_ids, [])

            log = output.getvalue()
            self.assertIn("mcp__dj__get_reaction_signal", log)
            self.assertIn("mcp__dj__narrate", log)
            self.assertIn("fake_camera", log)
        finally:
            await agent.disconnect()


if __name__ == "__main__":
    unittest.main()
