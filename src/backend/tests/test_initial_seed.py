"""Initial-seed resolution: env override > imported Redis key > default."""

import os
import unittest
from unittest import mock

from claude_dj.main import DEFAULT_INITIAL_SEED_TRACK_ID, resolve_initial_seed_track_id
from claude_dj.mcp.recommendations import (
    INITIAL_SEED_REDIS_KEY,
    RedisRecommendationClient,
    RedisRecommendationConfig,
)


class FakeSeedRedis:
    def __init__(self, get_value=None, raise_exc=None):
        self.get_value = get_value
        self.raise_exc = raise_exc
        self.commands = []

    def execute_command(self, *args):
        self.commands.append(args)
        if self.raise_exc is not None:
            raise self.raise_exc
        if args[0] == "GET":
            return self.get_value
        return None


def _client(fake):
    return RedisRecommendationClient(config=RedisRecommendationConfig(), client=fake)


class InitialSeedTest(unittest.TestCase):
    def test_get_returns_published_seed(self):
        client = _client(FakeSeedRedis(get_value=b"deezer:777"))
        self.assertEqual(client.get_initial_seed_track_id(), "deezer:777")

    def test_get_returns_none_when_unset(self):
        client = _client(FakeSeedRedis(get_value=None))
        self.assertIsNone(client.get_initial_seed_track_id())

    def test_get_degrades_to_none_on_redis_error(self):
        client = _client(FakeSeedRedis(raise_exc=OSError("down")))
        self.assertIsNone(client.get_initial_seed_track_id())

    def test_env_override_wins(self):
        client = _client(FakeSeedRedis(get_value=b"deezer:777"))
        with mock.patch.dict(os.environ, {"CLAUDE_DJ_INITIAL_REDIS_TRACK_ID": "deezer:override"}):
            self.assertEqual(resolve_initial_seed_track_id(client), "deezer:override")

    def test_imported_seed_used_when_no_override(self):
        client = _client(FakeSeedRedis(get_value=b"deezer:777"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_DJ_INITIAL_REDIS_TRACK_ID", None)
            self.assertEqual(resolve_initial_seed_track_id(client), "deezer:777")

    def test_default_when_nothing_published(self):
        client = _client(FakeSeedRedis(get_value=None))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_DJ_INITIAL_REDIS_TRACK_ID", None)
            self.assertEqual(resolve_initial_seed_track_id(client), DEFAULT_INITIAL_SEED_TRACK_ID)

    def test_redis_key_constant_matches_engine(self):
        self.assertEqual(INITIAL_SEED_REDIS_KEY, "claudedj:initial_seed_track_id")


if __name__ == "__main__":
    unittest.main()
