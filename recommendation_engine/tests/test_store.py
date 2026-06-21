"""Phase 4 acceptance: Redis index + track hashes + centroids.

These tests need a live Redis Stack (Search/Vector module). When Redis is
unreachable they SKIP gracefully — they never error or fail. When Redis is
present they assert:
  * indexed key count == number of input tracks
  * self-KNN of a stored vector returns that track at distance ~= 0
  * stored embedding byte length == 512 * 4 == 2048
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

import redis
from redis.commands.search.query import Query

from recommendation_engine import config, store_redis
from recommendation_engine.contracts import load_embeddings, load_enriched_tracks
from recommendation_engine.store_redis import EMBED_BYTES, EMBED_DTYPE


def _connect_or_skip() -> redis.Redis:
    """Return a live bytes-mode client, or skip the test if Redis is unreachable
    or lacks the Search module."""
    try:
        client = store_redis.get_redis_client()
        client.ping()
    except Exception as exc:  # connection refused, auth, timeout, DNS, etc.
        pytest.skip(f"Redis unreachable: {exc}")
    try:
        client.execute_command("FT._LIST")
    except Exception as exc:
        pytest.skip(f"Redis Search module unavailable: {exc}")
    return client


@pytest.fixture
def loaded_redis():
    """Load the committed fixtures into a uniquely-prefixed throwaway index,
    yield (client, index_name, tracks, embeddings), then clean up."""
    client = _connect_or_skip()

    suffix = uuid.uuid4().hex[:8]
    track_prefix = f"test_track_{suffix}:"
    centroid_prefix = f"test_centroid_{suffix}:"
    index_name = f"idx:test_tracks_{suffix}"

    # Patch config + module constants so the production loader writes to scratch keys.
    orig = (config.REDIS_INDEX, config.TRACK_KEY_PREFIX, config.CENTROID_KEY_PREFIX)
    config.REDIS_INDEX = index_name
    config.TRACK_KEY_PREFIX = track_prefix
    config.CENTROID_KEY_PREFIX = centroid_prefix

    tracks = load_enriched_tracks(config.FIXTURE_TRACKS_ENRICHED_PATH)
    embeddings = load_embeddings(config.FIXTURE_EMBEDDINGS_PATH)

    try:
        store_redis.load_into_redis(
            client,
            enriched_path=config.FIXTURE_TRACKS_ENRICHED_PATH,
            embeddings_path=config.FIXTURE_EMBEDDINGS_PATH,
        )
        yield client, index_name, track_prefix, centroid_prefix, tracks, embeddings
    finally:
        # Drop scratch keys + index.
        for t in tracks:
            client.delete(f"{track_prefix}{t.deezer_id}")
        genres = {t.genre_tag for t in tracks}
        for g in genres:
            client.delete(f"{centroid_prefix}{g}")
        try:
            client.ft(index_name).dropindex(delete_documents=False)
        except Exception:
            pass
        config.REDIS_INDEX, config.TRACK_KEY_PREFIX, config.CENTROID_KEY_PREFIX = orig


def test_indexed_key_count_matches_inputs(loaded_redis):
    client, index_name, track_prefix, _, tracks, _ = loaded_redis
    info = client.ft(index_name).info()
    num_docs = int(info["num_docs"])
    assert num_docs == len(tracks)


def test_stored_embedding_byte_length(loaded_redis):
    client, _, track_prefix, _, tracks, _ = loaded_redis
    for t in tracks:
        raw = client.hget(f"{track_prefix}{t.deezer_id}", "embedding")
        assert raw is not None
        assert len(raw) == EMBED_BYTES == 2048


def _raw_knn_top1(client, index_name, query_blob):
    """Issue a raw FT.SEARCH KNN and return (total, top_doc_id, top_score).

    redis-py 8.0's high-level ``ft().search()`` returns zero docs against a
    Redis 8 / RESP3 reply even when the index is populated, so we parse the
    raw command reply ourselves (handling both RESP2 lists and RESP3 dicts).
    """
    reply = client.execute_command(
        "FT.SEARCH", index_name,
        "*=>[KNN 1 @embedding $vec AS score]",
        "PARAMS", "2", "vec", query_blob,
        "SORTBY", "score", "ASC",
        "RETURN", "2", "score", "deezer_id",
        "DIALECT", "2",
    )

    def _d(v):
        return v.decode() if isinstance(v, (bytes, bytearray)) else v

    def _get(d, key):
        # Bytes-mode clients yield bytes keys; decode_responses clients yield str.
        if key in d:
            return d[key]
        return d.get(key.encode())

    if isinstance(reply, dict):  # RESP3
        total = _get(reply, "total_results") or 0
        results = _get(reply, "results") or []
        if not results:
            return total, None, None
        first = results[0]
        raw_attrs = _get(first, "extra_attributes") or {}
        attrs = {_d(k): _d(v) for k, v in raw_attrs.items()}
        return total, _d(_get(first, "id")), attrs.get("score")

    # RESP2: [total, key1, [f1, v1, ...], key2, ...]
    total = reply[0]
    if total < 1:
        return total, None, None
    doc_id = _d(reply[1])
    fields = {_d(reply[2][i]): _d(reply[2][i + 1]) for i in range(0, len(reply[2]), 2)}
    return total, doc_id, fields.get("score")


def test_self_knn_returns_same_track_at_zero_distance(loaded_redis):
    client, index_name, track_prefix, _, tracks, embeddings = loaded_redis
    vectors_by_id = {e["id"]: e["vector"] for e in embeddings}

    target = tracks[0]
    query_blob = np.asarray(vectors_by_id[target.id], dtype=EMBED_DTYPE).tobytes()

    total, doc_id, score = _raw_knn_top1(client, index_name, query_blob)

    assert total >= 1
    assert doc_id == f"{track_prefix}{target.deezer_id}"
    assert float(score) == pytest.approx(0.0, abs=1e-4)


def test_centroids_written(loaded_redis):
    client, _, _, centroid_prefix, tracks, _ = loaded_redis
    genres = {t.genre_tag for t in tracks}
    for g in genres:
        raw = client.hget(f"{centroid_prefix}{g}", "embedding")
        assert raw is not None
        assert len(raw) == EMBED_BYTES
        count = client.hget(f"{centroid_prefix}{g}", "count")
        assert int(count) >= 1
