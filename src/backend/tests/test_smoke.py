import io
import unittest

from claude_dj.mcp.narration import NarrationAudio
from claude_dj.mcp.playback import SpotifyDevice, SpotifyPlaybackState, SpotifyPlaylist, Track
from claude_dj.smoke import choose_starting_context, run_autonomous_demo


class FakeSpotifyPlayer:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.transferred: list[tuple[str, bool]] = []
        self.devices = [SpotifyDevice(id="device-1", name="MacBook", is_active=False, is_restricted=False)]
        self.state: SpotifyPlaybackState | None = None
        self.search_queries: list[str] = []
        self.playlists = [
            SpotifyPlaylist(id="playlist-1", name="baile inolvidable", total_tracks=10),
            SpotifyPlaylist(id="playlist-2", name="old skul", total_tracks=8),
        ]

    async def start_track(self, spotify_uri: str) -> None:
        self.started.append(spotify_uri)
        self.state = SpotifyPlaybackState(
            track_id="track-1",
            spotify_uri=spotify_uri,
            progress_ms=1_000,
            duration_ms=180_000,
            is_playing=True,
            device=SpotifyDevice(id="device-1", name="MacBook", is_active=True),
        )

    async def get_current_playback(self) -> SpotifyPlaybackState | None:
        return self.state

    async def search_tracks(self, query: str, limit: int = 6) -> list[Track]:
        self.search_queries.append(query)
        return [
            Track(
                id="track-1",
                title="Autonomous Track",
                artist="Demo Artist",
                spotify_uri="spotify:track:smoke",
                cluster=f"spotify_search:{query}",
            )
        ][:limit]

    async def list_user_playlists(self, limit: int = 20):
        return self.playlists[:limit]

    async def list_playlist_tracks(self, playlist_id: str, playlist_name: str, limit: int = 100) -> list[Track]:
        return []

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


class FakeNarrator:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def generate(self, text: str) -> NarrationAudio:
        self.texts.append(text)
        return NarrationAudio(
            id="narration-1",
            text=text,
            audio=b"audio-bytes",
            content_type="audio/mpeg",
            model="fake-model",
        )


class FakeAudioPlayer:
    def __init__(self) -> None:
        self.played: list[NarrationAudio] = []

    def play(self, narration: NarrationAudio) -> None:
        self.played.append(narration)


class SmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_choose_starting_context_uses_playlists_without_fixed_genre(self) -> None:
        spotify = FakeSpotifyPlayer()

        context = await choose_starting_context(spotify)

        self.assertEqual(context.query, "baile inolvidable old skul")
        self.assertIn("baile inolvidable", context.reason)

    async def test_autonomous_demo_runs_without_query_argument(self) -> None:
        spotify = FakeSpotifyPlayer()
        narrator = FakeNarrator()
        audio_player = FakeAudioPlayer()
        output = io.StringIO()

        result = await run_autonomous_demo(
            spotify=spotify,
            narrator=narrator,
            audio_player=audio_player,
            output=output,
        )

        self.assertEqual(result.track_id, "track-1")
        self.assertEqual(result.track_title, "Autonomous Track")
        self.assertEqual(result.starting_query, "baile inolvidable old skul")
        self.assertEqual(spotify.search_queries, ["baile inolvidable old skul"])
        self.assertEqual(spotify.transferred, [("device-1", False)])
        self.assertEqual(spotify.started, ["spotify:track:smoke"])
        self.assertEqual(len(narrator.texts), 1)
        self.assertIn("Autonomous Track", narrator.texts[0])
        self.assertIn("baile inolvidable", narrator.texts[0])
        self.assertEqual(audio_player.played[0].id, "narration-1")
        log = output.getvalue()
        self.assertIn("demo: autonomous start", log)
        self.assertIn("deepgram: ok played narration", log)
        self.assertIn("spotify: ok started Autonomous Track", log)
        self.assertIn("demo: ok", log)


if __name__ == "__main__":
    unittest.main()
