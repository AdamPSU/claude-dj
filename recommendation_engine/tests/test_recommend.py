"""Phase 5 acceptance: next_five() over a scratch Redis loaded from fixtures.

These tests require a Redis Stack (with the Search/Vector module) reachable via the
project's REDIS_* env vars (default localhost:6379). When Redis is unreachable or the
Search module is absent, every test SKIPS (never errors/fails) so the suite stays green
in environments without Redis (e.g. CI without a live instance).

The test owns its data: it populates a uniquely-prefixed scratch index + keys directly
from the committed fixtures (matching the Redis schema in engine_contracts.md), runs
next_five against that index, then tears everything down. It does NOT import store_redis.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from recommendation_engine import config, recommend
from recommendation_engine.contracts import load_embeddings, load_enriched_tracks

redis = pytest.importorskip("redis")
from redis.commands.search.field import TagField, TextField, VectorField  # noqa: E402
from redis.commands.search.index_definition import IndexDefinition, IndexType  # noqa: E402


# --- Scratch fixture: load committed fixtures into an isolated Redis namespace ----
@pytest.fixture()
def scratch_redis(monkeypatch):
    """Yield (client, prefix_info) with fixtures loaded; skip if Redis unavailable."""
    client = recommend.get_redis_client()
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001 - any connection error -> skip, never fail
        pytest.skip(f"Redis unreachable, skipping Phase 5 live tests: {exc}")

    token = uuid.uuid4().hex[:8]
    index_name = f"idx:test_tracks_{token}"
    track_prefix = f"test_track_{token}:"
    centroid_prefix = f"test_centroid_{token}:"

    # Point recommend.py's constants at this scratch namespace.
    monkeypatch.setattr(config, "REDIS_INDEX", index_name)
    monkeypatch.setattr(config, "TRACK_KEY_PREFIX", track_prefix)
    monkeypatch.setattr(config, "CENTROID_KEY_PREFIX", centroid_prefix)

    enriched = {t.id: t for t in load_enriched_tracks(config.FIXTURE_TRACKS_ENRICHED_PATH)}
    embeddings = {e["id"]: e["vector"] for e in load_embeddings(config.FIXTURE_EMBEDDINGS_PATH)}

    # Create the scratch index (FLAT cosine, matching the real schema).
    schema = (
        TagField("genre_tag"),
        TagField("artist"),
        TextField("title"),
        VectorField(
            "embedding",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": config.EMBED_DIM,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )
    definition = IndexDefinition(prefix=[track_prefix], index_type=IndexType.HASH)
    try:
        client.ft(index_name).create_index(schema, definition=definition)
    except redis.exceptions.ResponseError as exc:
        msg = str(exc).lower()
        if "unknown command" in msg or "ft.create" in msg:
            pytest.skip(f"Redis Search module not available: {exc}")
        raise

    # HSET each track with float32-bytes embedding.
    by_genre: dict[str, list[np.ndarray]] = {}
    for track_id, track in enriched.items():
        vec = np.asarray(embeddings[track_id], dtype=np.float32)
        assert vec.tobytes().__len__() == config.EMBED_DIM * 4
        deezer_id = track.deezer_id
        client.hset(
            f"{track_prefix}{deezer_id}",
            mapping={
                "title": track.title,
                "artist": track.artist,
                "album": track.album,
                "genre": track.genre,
                "genre_tag": track.genre_tag,
                "isrc": track.isrc,
                "deezer_id": track.deezer_id,
                "spotify_id": track.spotify_id,
                "artwork_url": track.artwork_url,
                "duration_seconds": str(track.duration_seconds),
                "rank": str(track.rank),
                "embedding": vec.tobytes(),
            },
        )
        by_genre.setdefault(track.genre_tag, []).append(vec)

    # Write L2-normalized per-genre centroids.
    for genre_tag, vecs in by_genre.items():
        mean = np.mean(np.stack(vecs), axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        mean = mean.astype(np.float32)
        client.hset(
            f"{centroid_prefix}{genre_tag}",
            mapping={
                "genre_tag": genre_tag,
                "count": str(len(vecs)),
                "embedding": mean.tobytes(),
            },
        )

    yield client

    # Teardown.
    try:
        client.ft(index_name).dropindex(delete_documents=False)
    except Exception:  # noqa: BLE001
        pass
    for track in enriched.values():
        client.delete(f"{track_prefix}{track.deezer_id}")
    for genre_tag in by_genre:
        client.delete(f"{centroid_prefix}{genre_tag}")


# Fixture corpus reference (see data/fixtures): two dance tracks + one distant pop track.
DANCE_A = "deezer:3135556"   # dance
DANCE_B = "deezer:14552159"  # dance
POP = "deezer:67238735"      # pop, distant from the dance pair


def _genre_tag_of(client, track_id: str) -> str:
    deezer_id = track_id.split(":", 1)[1]
    raw = client.hget(f"{config.TRACK_KEY_PREFIX}{deezer_id}", "genre_tag")
    return raw.decode() if isinstance(raw, bytes) else raw


# --- Tests -------------------------------------------------------------------
def test_positive_same_genre_excludes_current(scratch_redis):
    client = scratch_redis
    results = recommend.next_five(DANCE_A, "positive")
    assert DANCE_A not in results, "current track must never be returned"
    assert results, "expected at least one same-genre recommendation"
    for track_id in results:
        assert _genre_tag_of(client, track_id) == "dance"
    # Only one other dance track exists in the fixtures.
    assert results == [DANCE_B]


def test_neutral_behaves_like_positive(scratch_redis):
    client = scratch_redis
    results = recommend.next_five(DANCE_A, "neutral")
    assert DANCE_A not in results
    assert all(_genre_tag_of(client, t) == "dance" for t in results)


def test_negative_switches_to_most_distant_genre(scratch_redis):
    client = scratch_redis
    results = recommend.next_five(DANCE_A, "negative")
    assert results, "negative feedback should still return recommendations"
    assert DANCE_A not in results
    for track_id in results:
        assert _genre_tag_of(client, track_id) != "dance"
        assert _genre_tag_of(client, track_id) == "pop"
    assert results == [POP]


def test_recently_played_excluded(scratch_redis):
    # The only same-genre candidate for DANCE_A is DANCE_B; excluding it -> empty.
    results = recommend.next_five(DANCE_A, "positive", recently_played=[DANCE_B])
    assert DANCE_B not in results
    assert results == []


def test_invalid_signal_raises(scratch_redis):
    with pytest.raises(ValueError):
        recommend.next_five(DANCE_A, "sideways")
