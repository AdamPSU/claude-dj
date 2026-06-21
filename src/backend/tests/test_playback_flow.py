import unittest
import asyncio

from claude_dj.mcp.handlers import DJToolHandlers
from claude_dj.mcp.narration import NarrationAudio
from claude_dj.mcp.playback import InMemoryPlaybackRuntime, SpotifyDevice, SpotifyPlaybackState, SpotifyPlaylist, Track
from claude_dj.transition import InMemoryTransitionStore


class FakeNarrator:
    async def generate(self, text: str) -> NarrationAudio:
        return NarrationAudio(
            id="narration-1",
            text=text,
            audio=b"fake-audio",
            content_type="audio/mpeg",
            model="fake-model",
        )


class FakeReactionSource:
    def __init__(self) -> None:
        self.signal = {
            "available": True,
            "stub": False,
            "source": "fake_camera",
            "trend": "negative",
            "confidence": 0.97,
            "score": -0.92,
            "camera_state": "radical_change",
        }

    async def get_reaction_signal(self) -> dict[str, object]:
        return dict(self.signal)


class FakeSpotifyPlayer:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.state: SpotifyPlaybackState | None = None
        self.playlists: list[SpotifyPlaylist] = []
        self.playlist_tracks: dict[str, list[Track]] = {}
        self.search_results: list[Track] = []
        self.playlist_track_calls: list[tuple[str, int]] = []
        self.wait_for_parallel_fetch: asyncio.Event | None = None
        self.parallel_fetch_seen: asyncio.Event | None = None
        self.active_playlist_fetches = 0
        self.devices: list[SpotifyDevice] = []
        self.transferred: list[tuple[str, bool]] = []
        self.paused = 0
        self.resumed = 0
        self.volume_events: list[int] = []

    async def start_track(self, spotify_uri: str) -> None:
        self.started.append(spotify_uri)

    async def get_current_playback(self) -> SpotifyPlaybackState | None:
        return self.state

    async def search_tracks(self, query: str, limit: int = 6) -> list[Track]:
        return self.search_results[:limit]

    async def list_user_playlists(self, limit: int = 20) -> list[SpotifyPlaylist]:
        return self.playlists[:limit]

    async def list_playlist_tracks(self, playlist_id: str, playlist_name: str, limit: int = 100) -> list[Track]:
        self.playlist_track_calls.append((playlist_id, limit))
        self.active_playlist_fetches += 1
        try:
            if self.active_playlist_fetches >= 2 and self.parallel_fetch_seen:
                self.parallel_fetch_seen.set()
            if self.wait_for_parallel_fetch:
                await self.wait_for_parallel_fetch.wait()
            return self.playlist_tracks.get(playlist_id, [])[:limit]
        finally:
            self.active_playlist_fetches -= 1

    async def list_devices(self) -> list[SpotifyDevice]:
        return self.devices

    async def transfer_playback(self, device_id: str, *, play: bool = False) -> None:
        self.transferred.append((device_id, play))
        matching = next((device for device in self.devices if device.id == device_id), None)
        self.state = SpotifyPlaybackState(
            track_id=None,
            spotify_uri=None,
            progress_ms=0,
            duration_ms=0,
            is_playing=play,
            device=SpotifyDevice(
                id=device_id,
                name=matching.name if matching else "Spotify device",
                is_active=True,
                is_restricted=False,
            ),
        )

    async def pause_playback(self) -> None:
        self.paused += 1

    async def resume_playback(self) -> None:
        self.resumed += 1

    async def set_playback_volume(self, volume_percent: int) -> None:
        self.volume_events.append(volume_percent)


def build_handlers(spotify: FakeSpotifyPlayer) -> DJToolHandlers:
    runtime = InMemoryPlaybackRuntime(
        tracks=[
            Track(
                id="reggaeton-1",
                title="Reggaeton One",
                artist="Demo Artist",
                spotify_uri="spotify:track:reggaeton1",
                cluster="reggaeton",
                duration_ms=180_000,
            ),
            Track(
                id="reggaeton-2",
                title="Reggaeton Two",
                artist="Demo Artist",
                spotify_uri="spotify:track:reggaeton2",
                cluster="reggaeton",
                duration_ms=190_000,
            ),
            Track(
                id="smooth-1",
                title="Smooth One",
                artist="Demo Artist",
                spotify_uri="spotify:track:smooth1",
                cluster="smooth",
                duration_ms=200_000,
            ),
        ],
        spotify=spotify,
    )
    return DJToolHandlers(InMemoryTransitionStore(), FakeNarrator(), runtime)


class PlaybackFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_replace_queue_play_track_and_get_current_playback_use_app_owned_queue(self) -> None:
        spotify = FakeSpotifyPlayer()
        spotify.state = SpotifyPlaybackState(
            track_id="reggaeton-1",
            spotify_uri="spotify:track:reggaeton1",
            progress_ms=42_000,
            duration_ms=180_000,
            is_playing=True,
            device=SpotifyDevice(id="device-1", name="MacBook Pro", volume_percent=80),
        )
        handlers = build_handlers(spotify)

        replaced = await handlers.replace_queue(
            ["reggaeton-1", "reggaeton-2", "smooth-1"],
            reason="startup_set",
        )
        played = await handlers.play_track("reggaeton-1")
        playback = await handlers.get_current_playback()
        context = await handlers.get_session_context()

        self.assertEqual(replaced["queue_track_ids"], ["reggaeton-1", "reggaeton-2", "smooth-1"])
        self.assertFalse(replaced["stub"])
        self.assertEqual(spotify.started, ["spotify:track:reggaeton1"])
        self.assertEqual(played["track_id"], "reggaeton-1")
        self.assertFalse(played["stub"])
        self.assertEqual(playback["current_track"]["id"], "reggaeton-1")
        self.assertEqual(playback["current_track"]["title"], "Reggaeton One")
        self.assertEqual(playback["queue_track_ids"], ["reggaeton-2", "smooth-1"])
        self.assertEqual(playback["seconds_remaining"], 138)
        self.assertEqual(playback["cluster_streak"], 1)
        self.assertEqual(playback["device"]["name"], "MacBook Pro")
        self.assertEqual(context["current_track"]["id"], "reggaeton-1")
        self.assertEqual(context["queue_track_ids"], ["reggaeton-2", "smooth-1"])
        self.assertEqual(context["cluster_streak"], 1)

    async def test_demo_track_seconds_caps_effective_playback_duration(self) -> None:
        spotify = FakeSpotifyPlayer()
        spotify.state = SpotifyPlaybackState(
            track_id="reggaeton-1",
            spotify_uri="spotify:track:reggaeton1",
            progress_ms=42_000,
            duration_ms=180_000,
            is_playing=True,
            device=SpotifyDevice(id="device-1", name="MacBook Pro", volume_percent=80),
        )
        runtime = InMemoryPlaybackRuntime(
            tracks=[
                Track(
                    id="reggaeton-1",
                    title="Reggaeton One",
                    artist="Demo Artist",
                    spotify_uri="spotify:track:reggaeton1",
                    cluster="reggaeton",
                    duration_ms=180_000,
                )
            ],
            spotify=spotify,
            demo_track_seconds=30,
        )

        playback = await runtime.get_current_playback()

        self.assertEqual(playback["duration_ms"], 30_000)
        self.assertEqual(playback["progress_ms"], 30_000)
        self.assertEqual(playback["seconds_remaining"], 0)

    async def test_demo_track_seconds_counts_wall_clock_after_play_track(self) -> None:
        now = 100.0

        def clock() -> float:
            return now

        runtime = InMemoryPlaybackRuntime(
            tracks=[
                Track(
                    id="track-1",
                    title="One",
                    artist="Demo Artist",
                    spotify_uri="spotify:track:one",
                    cluster="demo",
                    duration_ms=180_000,
                )
            ],
            spotify=FakeSpotifyPlayer(),
            demo_track_seconds=20,
            clock=clock,
        )

        await runtime.play_track("track-1")
        now += 6.0
        playback = await runtime.get_current_playback()

        self.assertEqual(playback["progress_ms"], 6_000)
        self.assertEqual(playback["seconds_remaining"], 14)

    async def test_demo_track_seconds_excludes_paused_bridge_narration_time(self) -> None:
        now = 100.0

        def clock() -> float:
            return now

        runtime = InMemoryPlaybackRuntime(
            tracks=[
                Track(
                    id="track-1",
                    title="One",
                    artist="Demo Artist",
                    spotify_uri="spotify:track:one",
                    cluster="demo",
                    duration_ms=180_000,
                )
            ],
            spotify=FakeSpotifyPlayer(),
            demo_track_seconds=20,
            clock=clock,
        )

        await runtime.play_track("track-1")
        now += 1.0
        await runtime.pause_music()
        now += 10.0
        await runtime.resume_music()
        now += 5.0
        playback = await runtime.get_current_playback()

        self.assertEqual(playback["progress_ms"], 6_000)
        self.assertEqual(playback["seconds_remaining"], 14)

    async def test_replace_queue_after_current_track_sets_pending_queue(self) -> None:
        spotify = FakeSpotifyPlayer()
        handlers = build_handlers(spotify)

        await handlers.replace_queue(["reggaeton-1", "reggaeton-2"], reason="startup_set")
        result = await handlers.replace_queue(
            ["smooth-1"],
            reason="shift_after_negative_signal",
            timing="after_current_track",
        )

        playback = await handlers.get_current_playback()

        self.assertEqual(result["queue_track_ids"], ["reggaeton-1", "reggaeton-2"])
        self.assertEqual(result["pending_queue_track_ids"], ["smooth-1"])
        self.assertEqual(playback["queue_track_ids"], ["reggaeton-1", "reggaeton-2"])
        self.assertEqual(playback["pending_queue_track_ids"], ["smooth-1"])

    async def test_play_next_queued_track_promotes_pending_queue_at_boundary(self) -> None:
        spotify = FakeSpotifyPlayer()
        runtime = InMemoryPlaybackRuntime(
            tracks=[
                Track(id="current", title="Current", artist="Demo", spotify_uri="spotify:track:current", cluster="rap"),
                Track(id="next-1", title="Next One", artist="Demo", spotify_uri="spotify:track:next1", cluster="rap"),
                Track(id="next-2", title="Next Two", artist="Demo", spotify_uri="spotify:track:next2", cluster="rap"),
            ],
            spotify=spotify,
        )
        await runtime.play_track("current")
        await runtime.replace_queue(["next-1", "next-2"], reason="prepared_refill", timing="after_current_track")

        result = await runtime.play_next_queued_track()

        self.assertIsNotNone(result)
        self.assertEqual(result["track_id"], "next-1")
        self.assertEqual(spotify.started[-1], "spotify:track:next1")
        self.assertEqual(runtime.queue_track_ids, ["next-2"])
        self.assertEqual(runtime.pending_queue_track_ids, [])

    async def test_demo_queue_limit_caps_replacement_to_two_tracks(self) -> None:
        runtime = InMemoryPlaybackRuntime(
            tracks=[
                Track(id="track-1", title="One", artist="Demo", spotify_uri="spotify:track:1", cluster="rap"),
                Track(id="track-2", title="Two", artist="Demo", spotify_uri="spotify:track:2", cluster="rap"),
                Track(id="track-3", title="Three", artist="Demo", spotify_uri="spotify:track:3", cluster="rap"),
            ],
            queue_min_tracks=1,
            queue_max_tracks=2,
        )

        result = await runtime.replace_queue(["track-1", "track-2", "track-3"], reason="demo_cap")

        self.assertEqual(result["track_ids"], ["track-1", "track-2"])
        self.assertEqual(result["queue_track_ids"], ["track-1", "track-2"])
        self.assertEqual(result["dropped_track_ids"], ["track-3"])

    async def test_runtime_delegates_pause_and_resume_to_spotify(self) -> None:
        spotify = FakeSpotifyPlayer()
        runtime = InMemoryPlaybackRuntime(spotify=spotify)

        await runtime.pause_music()
        await runtime.resume_music()

        self.assertEqual(spotify.paused, 1)
        self.assertEqual(spotify.resumed, 1)

    async def test_runtime_reads_and_sets_spotify_music_volume(self) -> None:
        spotify = FakeSpotifyPlayer()
        spotify.state = SpotifyPlaybackState(
            track_id=None,
            spotify_uri=None,
            progress_ms=0,
            duration_ms=0,
            is_playing=True,
            device=SpotifyDevice(id="device-1", name="MacBook Pro", volume_percent=64),
        )
        runtime = InMemoryPlaybackRuntime(spotify=spotify)

        volume = await runtime.get_music_volume()
        await runtime.set_music_volume(42)

        self.assertEqual(volume, 64)
        self.assertEqual(spotify.volume_events, [42])

    async def test_get_reaction_signal_uses_injected_camera_source(self) -> None:
        handlers = DJToolHandlers(
            InMemoryTransitionStore(),
            FakeNarrator(),
            reaction_source=FakeReactionSource(),
        )

        signal = await handlers.get_reaction_signal()

        self.assertEqual(signal["source"], "fake_camera")
        self.assertEqual(signal["trend"], "negative")
        self.assertEqual(signal["camera_state"], "radical_change")
        self.assertGreater(signal["confidence"], 0.9)

    async def test_get_session_context_includes_spotify_playlist_names(self) -> None:
        spotify = FakeSpotifyPlayer()
        spotify.playlists = [
            SpotifyPlaylist(id="playlist-1", name="baile inolvidable"),
            SpotifyPlaylist(id="playlist-2", name="old skul"),
        ]
        runtime = InMemoryPlaybackRuntime(tracks=[], spotify=spotify)

        context = await runtime.get_session_context()

        self.assertEqual(context["seed_vibe"], "playlist-informed autonomous start")
        self.assertEqual(context["available_playlist_names"], ["baile inolvidable", "old skul"])

    async def test_play_track_promotes_pending_queue_when_transition_starts(self) -> None:
        spotify = FakeSpotifyPlayer()
        handlers = build_handlers(spotify)

        await handlers.replace_queue(["reggaeton-1", "reggaeton-2"], reason="startup_set")
        await handlers.replace_queue(["smooth-1"], reason="shift", timing="after_current_track")
        await handlers.play_track("smooth-1")
        playback = await handlers.get_current_playback()

        self.assertEqual(playback["current_track_id"], "smooth-1")
        self.assertEqual(playback["queue_track_ids"], [])
        self.assertEqual(playback["pending_queue_track_ids"], [])

    async def test_play_track_transfers_to_available_device_when_none_is_active(self) -> None:
        spotify = FakeSpotifyPlayer()
        spotify.devices = [SpotifyDevice(id="device-1", name="MacBook", is_active=False, is_restricted=False)]
        handlers = build_handlers(spotify)

        await handlers.replace_queue(["reggaeton-1"], reason="startup_set")
        await handlers.play_track("reggaeton-1")

        self.assertEqual(spotify.transferred, [("device-1", False)])
        self.assertEqual(spotify.started, ["spotify:track:reggaeton1"])

    async def test_play_track_reuses_preferred_device_for_later_tracks(self) -> None:
        spotify = FakeSpotifyPlayer()
        spotify.devices = [SpotifyDevice(id="device-1", name="MacBook", is_active=False, is_restricted=False)]
        handlers = build_handlers(spotify)

        await handlers.replace_queue(["reggaeton-1", "reggaeton-2"], reason="startup_set")
        await handlers.play_track("reggaeton-1")
        await handlers.play_track("reggaeton-2")

        self.assertEqual(spotify.transferred, [("device-1", False)])
        self.assertEqual(spotify.started, ["spotify:track:reggaeton1", "spotify:track:reggaeton2"])

    async def test_play_next_queued_track_starts_and_removes_first_queue_item(self) -> None:
        spotify = FakeSpotifyPlayer()
        runtime = InMemoryPlaybackRuntime(
            tracks=[
                Track(
                    id="reggaeton-1",
                    title="Reggaeton One",
                    artist="Demo Artist",
                    spotify_uri="spotify:track:reggaeton1",
                    cluster="reggaeton",
                ),
                Track(
                    id="reggaeton-2",
                    title="Reggaeton Two",
                    artist="Demo Artist",
                    spotify_uri="spotify:track:reggaeton2",
                    cluster="reggaeton",
                ),
            ],
            spotify=spotify,
        )

        await runtime.replace_queue(["reggaeton-1", "reggaeton-2"], reason="startup_set")
        first = await runtime.play_next_queued_track()
        second = await runtime.play_next_queued_track()
        third = await runtime.play_next_queued_track()

        self.assertEqual(first["track_id"], "reggaeton-1")
        self.assertEqual(second["track_id"], "reggaeton-2")
        self.assertIsNone(third)
        self.assertEqual(spotify.started, ["spotify:track:reggaeton1", "spotify:track:reggaeton2"])
        self.assertEqual((await runtime.get_current_playback())["queue_track_ids"], [])

    async def test_search_track_embeddings_uses_spotify_playlist_and_search_candidates(self) -> None:
        spotify = FakeSpotifyPlayer()
        spotify.playlists = [SpotifyPlaylist(id="playlist-1", name="Late Night Latin")]
        spotify.playlist_tracks = {
            "playlist-1": [
                Track(
                    id="playlist-track-1",
                    title="Latin Night Drive",
                    artist="Playlist Artist",
                    spotify_uri="spotify:track:playlist1",
                    cluster="playlist:Late Night Latin",
                )
            ]
        }
        spotify.search_results = [
            Track(
                id="search-track-1",
                title="Reggaeton Search Result",
                artist="Search Artist",
                spotify_uri="spotify:track:search1",
                cluster="spotify_search:latin night",
            )
        ]
        runtime = InMemoryPlaybackRuntime(tracks=[], spotify=spotify)

        result = await runtime.search_track_embeddings(query="latin night", limit=2)
        track_ids = [candidate["id"] for candidate in result["candidates"]]

        self.assertTrue(result["available"])
        self.assertFalse(result["stub"])
        self.assertEqual(result["source"], "spotify_playlist_search")
        self.assertTrue(result["temporary_until_embeddings"])
        self.assertEqual(track_ids, ["playlist-track-1", "search-track-1"])

        await runtime.replace_queue(track_ids, reason="startup_set")
        await runtime.play_track("playlist-track-1")

        self.assertEqual(spotify.started, ["spotify:track:playlist1"])

    async def test_search_track_embeddings_fetches_default_playlist_catalog_concurrently(self) -> None:
        spotify = FakeSpotifyPlayer()
        spotify.playlists = [
            SpotifyPlaylist(id=f"playlist-{index}", name=f"Playlist {index}")
            for index in range(1, 8)
        ]
        spotify.search_results = [
            Track(
                id="search-track-1",
                title="Search Track",
                artist="Search Artist",
                spotify_uri="spotify:track:search1",
                cluster="spotify_search:autonomous start",
            )
        ]
        spotify.wait_for_parallel_fetch = asyncio.Event()
        spotify.parallel_fetch_seen = asyncio.Event()
        runtime = InMemoryPlaybackRuntime(tracks=[], spotify=spotify)

        search_task = asyncio.create_task(runtime.search_track_embeddings(query="autonomous start", limit=3))
        await asyncio.wait_for(spotify.parallel_fetch_seen.wait(), timeout=1)
        spotify.wait_for_parallel_fetch.set()
        result = await search_task

        self.assertEqual(len(spotify.playlist_track_calls), 5)
        self.assertTrue(all(limit == 50 for _playlist_id, limit in spotify.playlist_track_calls))
        self.assertEqual(result["candidates"][0]["id"], "search-track-1")


if __name__ == "__main__":
    unittest.main()
