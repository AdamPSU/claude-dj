import unittest
from array import array

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


class MultiGenreFakeRedis:
    def __init__(self) -> None:
        self.seed_vector = array("f", [1.0, 0.0]).tobytes()
        self.hashes = {
            "track:seed": {
                b"embedding": self.seed_vector,
                b"genre_tag": b"rap_hip_hop",
                b"spotify_id": b"seed",
                b"title": b"Seed",
                b"artist": b"Seed Artist",
                b"duration_seconds": b"180",
            }
        }
        self.centroids = {
            "genre_centroid:singer_songwriter": array("f", [-1.0, 0.0]).tobytes(),
            "genre_centroid:salsa": array("f", [0.0, 1.0]).tobytes(),
        }
        self.rows_by_genre = {
            "singer_songwriter": [
                self._row("track:singer1", "Singer One", "Singer Artist", "singer_songwriter", "singer1", 10),
                self._row("track:singer2", "Singer Two", "Singer Artist", "singer_songwriter", "singer2", 9),
            ],
            "salsa": [
                self._row("track:salsa1", "Salsa One", "Salsa Artist", "salsa", "salsa1", 8),
                self._row("track:salsa2", "Salsa Two", "Salsa Artist", "salsa", "salsa2", 7),
            ],
        }

    def hgetall(self, key):
        return self.hashes.get(key, {})

    def hget(self, key, field):
        if field == "embedding":
            return self.centroids.get(key)
        return None

    def execute_command(self, *args):
        if args[0] == "KEYS":
            return list(self.centroids)
        if args[0] == "FT.SEARCH" and "KNN" in args[2]:
            genre = args[2].split("{", 1)[1].split("}", 1)[0]
            rows = self.rows_by_genre.get(genre, [])
            response = [len(rows)]
            for doc_id, fields in rows:
                response.extend([doc_id.encode(), fields])
            return response
        raise AssertionError(f"unexpected Redis command: {args}")

    @staticmethod
    def _row(track_key, title, artist, genre, spotify_id, rank):
        return (
            track_key,
            [
                b"score",
                b"0.1",
                b"rank",
                str(rank).encode(),
                b"title",
                title.encode(),
                b"artist",
                artist.encode(),
                b"genre_tag",
                genre.encode(),
                b"spotify_id",
                spotify_id.encode(),
                b"duration_seconds",
                b"180",
                b"artwork_url",
                b"https://image.example/cover.jpg",
            ],
        )


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

    async def test_negative_shift_backfills_from_next_distant_genre(self) -> None:
        recommendations = RedisRecommendationClient(client=MultiGenreFakeRedis())

        result = await recommendations.recommend(
            seed_track_id="deezer:seed",
            signal="negative",
            mode="shift",
            limit=3,
            avoid_clusters=["rap_hip_hop"],
            exclude_track_ids=[],
        )

        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(
            [candidate.cluster for candidate in result.candidates],
            ["singer_songwriter", "singer_songwriter", "salsa"],
        )
        self.assertEqual(result.target_genre, "singer_songwriter")


class RedisConfigTests(unittest.TestCase):
    def test_client_kwargs_use_resp3_for_hello_auth(self) -> None:
        config = RedisRecommendationConfig(host="example.com", port=18497, username="default", password="secret")

        self.assertEqual(config.client_kwargs()["protocol"], 3)
        self.assertFalse(config.client_kwargs()["maint_notifications_config"].enabled)


if __name__ == "__main__":
    unittest.main()
