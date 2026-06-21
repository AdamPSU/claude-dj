import unittest

from claude_dj.mcp.handlers import DJToolHandlers
from claude_dj.mcp.narration import NarrationAudio
from claude_dj.mcp.playback import InMemoryPlaybackRuntime, SpotifyPlayer
from claude_dj.mcp.recommendations import RedisRecommendationClient, RedisRecommendationConfig
from claude_dj.transition import InMemoryTransitionStore


class FakeRedis:
    def __init__(self) -> None:
        self.commands: list[tuple[object, ...]] = []
        self.hashes = {
            "track:100814018": {
                b"embedding": b"seed-vector",
                b"genre_tag": b"rap_hip_hop",
                b"deezer_id": b"100814018",
                b"spotify_id": b"4RY96Asd9IefaL3X4LOLZ8",
                b"title": b"Don't",
                b"artist": b"Bryson Tiller",
                b"duration_seconds": b"198",
                b"artwork_url": b"https://image.example/seed.jpg",
                b"rank": b"123",
            },
            "track:434211382": {
                b"embedding": b"next-vector",
                b"genre_tag": b"r_b",
                b"deezer_id": b"434211382",
                b"spotify_id": b"0sBJA2OCEECMs0HsdIQhvR",
                b"title": b"Sky Walker (feat. Travis Scott)",
                b"artist": b"Miguel",
                b"duration_seconds": b"259",
                b"artwork_url": b"https://image.example/sky.jpg",
                b"rank": b"100",
            },
        }

    def hgetall(self, key):
        return self.hashes.get(key, {})

    def execute_command(self, *args):
        self.commands.append(args)
        if args[0] == "FT.SEARCH" and args[2] == "*":
            return [
                2,
                b"track:100814018",
                [b"title", b"Don't", b"artist", b"Bryson Tiller", b"genre_tag", b"rap_hip_hop", b"spotify_id", b"4RY96Asd9IefaL3X4LOLZ8", b"duration_seconds", b"198", b"artwork_url", b"https://image.example/seed.jpg", b"rank", b"123", b"deezer_id", b"100814018"],
                b"track:434211382",
                [b"title", b"Sky Walker (feat. Travis Scott)", b"artist", b"Miguel", b"genre_tag", b"r_b", b"spotify_id", b"0sBJA2OCEECMs0HsdIQhvR", b"duration_seconds", b"259", b"artwork_url", b"https://image.example/sky.jpg", b"rank", b"100", b"deezer_id", b"434211382"],
            ]
        if args[0] == "FT.SEARCH" and "KNN" in args[2]:
            return [
                1,
                b"track:434211382",
                [
                    b"score",
                    b"0.1243",
                    b"rank",
                    b"100",
                    b"title",
                    b"Sky Walker (feat. Travis Scott)",
                    b"artist",
                    b"Miguel",
                    b"genre_tag",
                    b"r_b",
                    b"spotify_id",
                    b"0sBJA2OCEECMs0HsdIQhvR",
                    b"duration_seconds",
                    b"259",
                    b"artwork_url",
                    b"https://image.example/sky.jpg",
                ],
            ]
        raise AssertionError(f"unexpected Redis command: {args}")


class FakeSpotify(SpotifyPlayer):
    def __init__(self) -> None:
        self.started: list[str] = []
        self.search_queries: list[str] = []

    async def start_track(self, spotify_uri: str) -> None:
        self.started.append(spotify_uri)

    async def get_current_playback(self):
        return None

    async def search_tracks(self, query: str, limit: int = 6):
        self.search_queries.append(query)
        return []

    async def list_user_playlists(self, limit: int = 20):
        return []

    async def list_playlist_tracks(self, playlist_id: str, playlist_name: str, limit: int = 100):
        return []

    async def list_devices(self):
        return []

    async def transfer_playback(self, device_id: str, *, play: bool = False) -> None:
        return None


class FakeNarrator:
    async def generate(self, text: str) -> NarrationAudio:
        return NarrationAudio(
            id="narration-1",
            text=text,
            audio=b"fake-audio",
            content_type="audio/mpeg",
            model="fake-model",
        )


class RecommendationBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_seed_candidates_derive_deezer_ids_from_track_keys(self) -> None:
        recommendations = RedisRecommendationClient(client=FakeRedis())

        candidates = await recommendations.seed_candidates(limit=2, avoid_clusters=[])

        self.assertEqual([candidate.id for candidate in candidates], ["deezer:100814018", "deezer:434211382"])
        self.assertEqual(candidates[0].title, "Don't")
        self.assertEqual(candidates[0].cluster, "rap_hip_hop")

    async def test_search_registers_redis_tracks_so_queue_and_playback_work(self) -> None:
        redis = FakeRedis()
        spotify = FakeSpotify()
        runtime = InMemoryPlaybackRuntime(
            tracks=[],
            spotify=spotify,
            recommendations=RedisRecommendationClient(client=redis),
            initial_seed_track_id="deezer:100814018",
            require_recommendations=True,
        )

        result = await runtime.search_track_embeddings(seed_track_id="deezer:100814018", signal="positive", limit=1)
        track_ids = [candidate["id"] for candidate in result["candidates"]]

        self.assertEqual(result["source"], "redis_vector")
        self.assertEqual(result["seed_track_id"], "deezer:100814018")
        self.assertEqual(track_ids, ["deezer:434211382"])
        self.assertEqual(spotify.search_queries, [])

        await runtime.replace_queue(track_ids, reason="redis_recommendation")
        await runtime.play_track("deezer:434211382")

        self.assertEqual(spotify.started, ["spotify:track:0sBJA2OCEECMs0HsdIQhvR"])

    async def test_handler_accepts_seed_and_signal_arguments(self) -> None:
        runtime = InMemoryPlaybackRuntime(
            tracks=[],
            spotify=FakeSpotify(),
            recommendations=RedisRecommendationClient(client=FakeRedis()),
            initial_seed_track_id="deezer:100814018",
            require_recommendations=True,
        )
        handlers = DJToolHandlers(InMemoryTransitionStore(), FakeNarrator(), runtime)

        result = await handlers.search_track_embeddings(seed_track_id="deezer:100814018", signal="positive", limit=1)

        self.assertEqual(result["candidates"][0]["id"], "deezer:434211382")


class RedisConfigTests(unittest.TestCase):
    def test_client_kwargs_use_resp3_for_hello_auth(self) -> None:
        config = RedisRecommendationConfig(host="example.com", port=18497, username="default", password="secret")

        self.assertEqual(config.client_kwargs()["protocol"], 3)
        self.assertFalse(config.client_kwargs()["maint_notifications_config"].enabled)


if __name__ == "__main__":
    unittest.main()
