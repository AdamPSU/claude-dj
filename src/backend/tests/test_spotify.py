import json
import socket
import unittest
from unittest.mock import patch

from claude_dj.mcp import spotify as spotify_module
from claude_dj.mcp.spotify import SpotifyConfig, SpotifyWebAPIPlayer


class FakeResponse:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        if self.payload is None:
            return b""
        return json.dumps(self.payload).encode("utf-8")


class FakeRawResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class SpotifyWebAPIPlayerTests(unittest.IsolatedAsyncioTestCase):
    def test_ipv4_connection_resolver_only_asks_for_ipv4_addresses(self) -> None:
        calls = []

        class FakeSocket:
            def __init__(self, family, socktype, proto) -> None:
                self.family = family
                self.socktype = socktype
                self.proto = proto
                self.timeout = None
                self.connected_to = None
                self.closed = False

            def settimeout(self, timeout) -> None:
                self.timeout = timeout

            def connect(self, address) -> None:
                self.connected_to = address

            def close(self) -> None:
                self.closed = True

        def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            calls.append((host, port, family, type, proto, flags))
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("35.186.224.24", 443))]

        created_sockets = []

        def fake_socket(family, socktype, proto):
            sock = FakeSocket(family, socktype, proto)
            created_sockets.append(sock)
            return sock

        with patch("claude_dj.mcp.spotify.socket.getaddrinfo", fake_getaddrinfo), patch(
            "claude_dj.mcp.spotify.socket.socket",
            fake_socket,
        ):
            sock = spotify_module.create_ipv4_connection(("api.spotify.com", 443), timeout=1.5)

        self.assertEqual(calls, [("api.spotify.com", 443, socket.AF_INET, socket.SOCK_STREAM, 0, 0)])
        self.assertIs(sock, created_sockets[0])
        self.assertEqual(sock.timeout, 1.5)
        self.assertEqual(sock.connected_to, ("35.186.224.24", 443))
        self.assertFalse(sock.closed)

    async def test_start_track_refreshes_token_and_puts_track_uri_to_active_player(self) -> None:
        requests = []
        timeouts = []

        def fake_urlopen(request, timeout=None):
            requests.append(request)
            timeouts.append(timeout)
            if request.full_url == "https://accounts.spotify.com/api/token":
                return FakeResponse({"access_token": "access-token-1"})
            return FakeResponse()

        player = SpotifyWebAPIPlayer(
            SpotifyConfig(
                client_id="client-id",
                client_secret="client-secret",
                refresh_token="refresh-token",
                request_timeout_seconds=7.0,
            )
        )

        with patch("claude_dj.mcp.spotify.urlopen", fake_urlopen):
            await player.start_track("spotify:track:abc")

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].full_url, "https://accounts.spotify.com/api/token")
        self.assertIn(b"grant_type=refresh_token", requests[0].data)
        self.assertIn(b"refresh_token=refresh-token", requests[0].data)
        self.assertEqual(
            requests[1].full_url,
            "https://api.spotify.com/v1/me/player/play",
        )
        self.assertEqual(requests[1].headers["Authorization"], "Bearer access-token-1")
        self.assertEqual(json.loads(requests[1].data.decode("utf-8")), {"uris": ["spotify:track:abc"]})
        self.assertEqual(timeouts, [7.0, 7.0])

    async def test_pause_and_resume_call_spotify_player_endpoints(self) -> None:
        responses = [FakeResponse({"access_token": "access-token-1"}), FakeResponse(), FakeResponse()]
        requests = []

        def fake_urlopen(request, timeout=None):
            requests.append(request)
            return responses.pop(0)

        player = SpotifyWebAPIPlayer(
            SpotifyConfig(
                client_id="client-id",
                client_secret="client-secret",
                refresh_token="refresh-token",
            )
        )

        with patch("claude_dj.mcp.spotify.urlopen", fake_urlopen):
            await player.pause_playback()
            await player.resume_playback()

        self.assertEqual(requests[1].full_url, "https://api.spotify.com/v1/me/player/pause")
        self.assertEqual(requests[2].full_url, "https://api.spotify.com/v1/me/player/play")
        self.assertIsNone(requests[1].data)
        self.assertIsNone(requests[2].data)

    async def test_set_playback_volume_calls_spotify_volume_endpoint(self) -> None:
        responses = [FakeResponse({"access_token": "access-token-1"}), FakeResponse()]
        requests = []

        def fake_urlopen(request, timeout=None):
            requests.append(request)
            return responses.pop(0)

        player = SpotifyWebAPIPlayer(
            SpotifyConfig(
                client_id="client-id",
                client_secret="client-secret",
                refresh_token="refresh-token",
            )
        )

        with patch("claude_dj.mcp.spotify.urlopen", fake_urlopen):
            await player.set_playback_volume(42)

        self.assertEqual(requests[1].full_url, "https://api.spotify.com/v1/me/player/volume?volume_percent=42")
        self.assertIsNone(requests[1].data)

    async def test_resume_playback_accepts_non_json_success_body(self) -> None:
        responses = [FakeResponse({"access_token": "access-token-1"}), FakeRawResponse(b"OK")]

        def fake_urlopen(request, timeout=None):
            return responses.pop(0)

        player = SpotifyWebAPIPlayer(
            SpotifyConfig(
                client_id="client-id",
                client_secret="client-secret",
                refresh_token="refresh-token",
            )
        )

        with patch("claude_dj.mcp.spotify.urlopen", fake_urlopen):
            await player.resume_playback()

    async def test_search_tracks_maps_spotify_response_to_runtime_tracks(self) -> None:
        responses = [
            FakeResponse({"access_token": "access-token-1"}),
            FakeResponse(
                {
                    "tracks": {
                        "items": [
                            {
                                "id": "spotify-track-id",
                                "uri": "spotify:track:abc",
                                "name": "Late Night Reggaeton",
                                "artists": [{"name": "Demo Artist"}],
                                "album": {"images": [{"url": "https://image.example/art.jpg"}]},
                                "duration_ms": 181_000,
                                "is_local": False,
                            }
                        ]
                    }
                }
            ),
        ]
        requests = []

        def fake_urlopen(request, timeout=None):
            requests.append(request)
            return responses.pop(0)

        player = SpotifyWebAPIPlayer(
            SpotifyConfig(
                client_id="client-id",
                client_secret="client-secret",
                refresh_token="refresh-token",
            )
        )

        with patch("claude_dj.mcp.spotify.urlopen", fake_urlopen):
            tracks = await player.search_tracks("reggaeton night", limit=3)

        self.assertEqual(
            requests[1].full_url,
            "https://api.spotify.com/v1/search?q=reggaeton+night&type=track&limit=3",
        )
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].id, "spotify-track-id")
        self.assertEqual(tracks[0].title, "Late Night Reggaeton")
        self.assertEqual(tracks[0].artist, "Demo Artist")
        self.assertEqual(tracks[0].spotify_uri, "spotify:track:abc")
        self.assertEqual(tracks[0].cluster, "spotify_search:reggaeton night")
        self.assertEqual(tracks[0].artwork_url, "https://image.example/art.jpg")

    async def test_search_tracks_caps_limit_for_spotify_api(self) -> None:
        responses = [
            FakeResponse({"access_token": "access-token-1"}),
            FakeResponse({"tracks": {"items": []}}),
        ]
        requests = []

        def fake_urlopen(request, timeout=None):
            requests.append(request)
            return responses.pop(0)

        player = SpotifyWebAPIPlayer(
            SpotifyConfig(
                client_id="client-id",
                client_secret="client-secret",
                refresh_token="refresh-token",
            )
        )

        with patch("claude_dj.mcp.spotify.urlopen", fake_urlopen):
            await player.search_tracks("warm opener", limit=12)

        self.assertEqual(
            requests[1].full_url,
            "https://api.spotify.com/v1/search?q=warm+opener&type=track&limit=10",
        )

    async def test_user_playlist_tracks_skip_local_tracks_and_map_metadata(self) -> None:
        responses = [
            FakeResponse({"access_token": "access-token-1"}),
            FakeResponse(
                {
                    "items": [
                        {
                            "id": "playlist-1",
                            "name": "Late Night Latin",
                            "public": False,
                            "collaborative": True,
                            "tracks": {"total": 42},
                        }
                    ]
                }
            ),
            FakeResponse(
                {
                    "items": [
                        {
                            "track": {
                                "id": "playlist-track-id",
                                "uri": "spotify:track:def",
                                "name": "Playlist Track",
                                "artists": [{"name": "Playlist Artist"}],
                                "album": {"images": [{"url": "https://image.example/playlist.jpg"}]},
                                "duration_ms": 199_000,
                                "is_local": False,
                            }
                        },
                        {"track": {"id": "local-track", "name": "Local Track", "is_local": True}},
                    ]
                }
            ),
        ]
        requests = []

        def fake_urlopen(request, timeout=None):
            requests.append(request)
            return responses.pop(0)

        player = SpotifyWebAPIPlayer(
            SpotifyConfig(
                client_id="client-id",
                client_secret="client-secret",
                refresh_token="refresh-token",
            )
        )

        with patch("claude_dj.mcp.spotify.urlopen", fake_urlopen):
            playlists = await player.list_user_playlists(limit=20)
            tracks = await player.list_playlist_tracks(playlists[0].id, playlists[0].name, limit=100)

        self.assertEqual(requests[1].full_url, "https://api.spotify.com/v1/me/playlists?limit=20")
        self.assertEqual(requests[2].full_url, "https://api.spotify.com/v1/playlists/playlist-1/items?limit=100")
        self.assertEqual(playlists[0].id, "playlist-1")
        self.assertEqual(playlists[0].name, "Late Night Latin")
        self.assertFalse(playlists[0].public)
        self.assertTrue(playlists[0].collaborative)
        self.assertEqual(playlists[0].total_tracks, 42)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].id, "playlist-track-id")
        self.assertEqual(tracks[0].cluster, "playlist:Late Night Latin")

    async def test_get_current_playback_maps_spotify_response(self) -> None:
        responses = [
            FakeResponse({"access_token": "access-token-1"}),
            FakeResponse(
                {
                    "is_playing": True,
                    "progress_ms": 12_000,
                    "device": {"id": "device-1", "name": "MacBook", "volume_percent": 77},
                    "item": {
                        "id": "spotify-track-id",
                        "uri": "spotify:track:abc",
                        "duration_ms": 180_000,
                    },
                }
            ),
        ]

        def fake_urlopen(request, timeout=None):
            return responses.pop(0)

        player = SpotifyWebAPIPlayer(
            SpotifyConfig(
                client_id="client-id",
                client_secret="client-secret",
                refresh_token="refresh-token",
            )
        )

        with patch("claude_dj.mcp.spotify.urlopen", fake_urlopen):
            state = await player.get_current_playback()

        self.assertEqual(state.track_id, "spotify-track-id")
        self.assertEqual(state.spotify_uri, "spotify:track:abc")
        self.assertEqual(state.progress_ms, 12_000)
        self.assertEqual(state.duration_ms, 180_000)
        self.assertTrue(state.is_playing)
        self.assertEqual(state.device.name, "MacBook")
        self.assertEqual(state.device.volume_percent, 77)

    async def test_refreshes_access_token_after_expiry(self) -> None:
        now = [0.0]
        responses = [
            FakeResponse({"access_token": "access-token-1", "expires_in": 3600}),
            FakeResponse({"is_playing": False, "item": None}),
            FakeResponse({"access_token": "access-token-2", "expires_in": 3600}),
            FakeResponse({"is_playing": False, "item": None}),
        ]
        requests = []

        def fake_urlopen(request, timeout=None):
            requests.append(request)
            return responses.pop(0)

        player = SpotifyWebAPIPlayer(
            SpotifyConfig(
                client_id="client-id",
                client_secret="client-secret",
                refresh_token="refresh-token",
            ),
            monotonic=lambda: now[0],
        )

        with patch("claude_dj.mcp.spotify.urlopen", fake_urlopen):
            await player.get_current_playback()
            now[0] = 4_000.0
            await player.get_current_playback()

        self.assertEqual(requests[0].full_url, "https://accounts.spotify.com/api/token")
        self.assertEqual(requests[1].headers["Authorization"], "Bearer access-token-1")
        self.assertEqual(requests[2].full_url, "https://accounts.spotify.com/api/token")
        self.assertEqual(requests[3].headers["Authorization"], "Bearer access-token-2")


if __name__ == "__main__":
    unittest.main()
