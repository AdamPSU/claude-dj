"""Import-history acceptance: provider -> Deezer -> CLAP -> Redis -> seed.

Every dependency is injected/faked: no network, no CLAP/torch, no live Redis.
The Deezer seam reuses the same fake-client pattern as test_enrich.
"""

from __future__ import annotations

import pytest

from recommendation_engine import config
from recommendation_engine.enrich_deezer import DeezerClient, DeezerError
from recommendation_engine.import_history import import_last_played
from recommendation_engine.import_history.provider import ExternalTrack
from recommendation_engine.import_history.spotify_provider import SpotifyHistoryProvider

EMBED_DIM = config.EMBED_DIM
PREVIEW_BYTES = b"ID3-fake-mp3-bytes"


# --- Fakes -------------------------------------------------------------------
def _deezer_track(deezer_id, album_id, *, preview=True, title="Song", artist="Artist"):
    return {
        "id": deezer_id,
        "title": title,
        "duration": 226,
        "rank": 814839,
        "preview": "https://cdn.deezer.com/preview.mp3" if preview else "",
        "artist": {"name": artist},
        "album": {"id": album_id, "title": "Album", "cover_xl": "https://art/xl.jpg"},
    }


class FakeDeezer(DeezerClient):
    """DeezerClient with both network seams served from dicts (see test_enrich)."""

    def __init__(self, isrc_map=None, album_map=None):
        super().__init__(sleep=lambda _s: None)
        self.isrc_map = isrc_map or {}
        self.album_map = album_map or {}

    def get_json(self, path, params=None):
        if path.startswith("/track/isrc:"):
            isrc = path.split("/track/isrc:")[1]
            val = self.isrc_map.get(isrc)
            if val is None:
                raise DeezerError("not found")
            return val
        if path == "/search":
            return {"data": []}
        if path.startswith("/album/"):
            album_id = path.split("/album/")[1]
            return self.album_map.get(album_id, {"genres": {"data": []}})
        raise DeezerError(f"unexpected path {path}")

    def download(self, url):
        return PREVIEW_BYTES


class FakeRedis:
    """Captures store_track HSETs, the published seed SET, and TTLs."""

    def __init__(self):
        self.hashes: dict[str, dict] = {}
        self.kv: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def hset(self, key, mapping=None):
        self.hashes.setdefault(key, {}).update(mapping or {})

    def set(self, key, value, ex=None):
        self.kv[key] = value
        if ex is not None:
            self.ttls[key] = ex

    def expire(self, key, seconds):
        self.ttls[key] = seconds


class FakeProvider:
    name = "spotify"

    def __init__(self, ext):
        self._ext = ext

    def last_played(self) -> ExternalTrack | None:
        return self._ext


def _ext(isrc="USRC11500001"):
    return ExternalTrack(
        title="Song",
        artist="Artist",
        isrc=isrc,
        album_name="Album",
        source="spotify",
        source_id="spotifyid123",
    )


def _embed_fn(_path):
    # Unit vector of the right dimension; store_track normalizes/encodes it.
    vec = [0.0] * EMBED_DIM
    vec[0] = 1.0
    return vec


def _import(provider, deezer, redis, *, recommend_fn, audio_dir):
    return import_last_played(
        provider,
        deezer_client=deezer,
        embed_fn=_embed_fn,
        recommend_fn=recommend_fn,
        redis_client=redis,
        audio_dir=audio_dir,
        log=lambda *_a: None,
    )


# --- Tests -------------------------------------------------------------------
def test_successful_import_stores_track_and_publishes_seed(tmp_path):
    deezer = FakeDeezer(
        isrc_map={"USRC11500001": _deezer_track("555", "900")},
        album_map={"900": {"genres": {"data": [{"name": "Rap/Hip Hop"}]}}},
    )
    redis = FakeRedis()
    result = _import(
        FakeProvider(_ext()), deezer, redis,
        recommend_fn=lambda _id, _sig: ["deezer:1", "deezer:2"],
        audio_dir=tmp_path,
    )

    assert result.imported is True
    assert result.fell_back is False
    assert result.seed_track_id == "deezer:555"
    assert result.source == "spotify"
    # Stored under the standard track key with genre + the carried-through id.
    stored = redis.hashes["track:555"]
    assert stored["genre_tag"] == "rap_hip_hop"
    assert stored["spotify_id"] == "spotifyid123"
    assert "embedding" in stored
    # Seed published for the harness.
    assert redis.kv[config.INITIAL_SEED_REDIS_KEY] == "deezer:555"
    # Ephemeral: both the track hash and the seed pointer carry a 1h TTL.
    assert redis.ttls["track:555"] == 3600
    assert redis.ttls[config.INITIAL_SEED_REDIS_KEY] == 3600


def test_no_history_falls_back_to_default(tmp_path):
    redis = FakeRedis()
    result = _import(
        FakeProvider(None), FakeDeezer(), redis,
        recommend_fn=lambda _id, _sig: ["deezer:1"],
        audio_dir=tmp_path,
    )
    assert result.fell_back is True
    assert result.reason == "no_history"
    assert result.seed_track_id == config.DEFAULT_SEED_TRACK_ID
    assert redis.kv[config.INITIAL_SEED_REDIS_KEY] == config.DEFAULT_SEED_TRACK_ID


def test_no_genre_falls_back_to_default(tmp_path):
    deezer = FakeDeezer(
        isrc_map={"USRC11500001": _deezer_track("555", "900")},
        album_map={"900": {"genres": {"data": []}}},  # album has no genre
    )
    redis = FakeRedis()
    result = _import(
        FakeProvider(_ext()), deezer, redis,
        recommend_fn=lambda _id, _sig: ["deezer:1"],
        audio_dir=tmp_path,
    )
    assert result.fell_back is True
    assert result.reason == "no_genre"
    assert result.seed_track_id == config.DEFAULT_SEED_TRACK_ID
    assert "track:555" not in redis.hashes  # never stored


def test_no_deezer_match_falls_back_to_default(tmp_path):
    deezer = FakeDeezer(isrc_map={}, album_map={})  # ISRC + search both miss
    redis = FakeRedis()
    result = _import(
        FakeProvider(_ext()), deezer, redis,
        recommend_fn=lambda _id, _sig: ["deezer:1"],
        audio_dir=tmp_path,
    )
    assert result.fell_back is True
    assert result.reason == "no_match"
    assert result.seed_track_id == config.DEFAULT_SEED_TRACK_ID


def test_empty_genre_pool_falls_back_but_keeps_stored_track(tmp_path):
    deezer = FakeDeezer(
        isrc_map={"USRC11500001": _deezer_track("555", "900")},
        album_map={"900": {"genres": {"data": [{"name": "Rap/Hip Hop"}]}}},
    )
    redis = FakeRedis()
    result = _import(
        FakeProvider(_ext()), deezer, redis,
        recommend_fn=lambda _id, _sig: [],  # genre too sparse to recommend
        audio_dir=tmp_path,
    )
    assert result.fell_back is True
    assert result.reason == "no_candidates_in_genre"
    assert result.seed_track_id == config.DEFAULT_SEED_TRACK_ID
    assert "track:555" in redis.hashes  # track was still stored/indexed
    assert redis.kv[config.INITIAL_SEED_REDIS_KEY] == config.DEFAULT_SEED_TRACK_ID


def test_spotify_provider_parses_recently_played_track():
    track = {
        "id": "abc",
        "name": "Don't",
        "artists": [{"name": "Bryson Tiller"}],
        "external_ids": {"isrc": "usrc11501051"},
        "album": {"name": "T R A P S O U L"},
    }
    ext = SpotifyHistoryProvider._to_external(track)
    assert ext == ExternalTrack(
        title="Don't",
        artist="Bryson Tiller",
        isrc="USRC11501051",  # upper-cased
        album_name="T R A P S O U L",
        source="spotify",
        source_id="abc",
    )


def test_spotify_provider_rejects_unusable_track():
    assert SpotifyHistoryProvider._to_external({"name": "x", "artists": []}) is None
