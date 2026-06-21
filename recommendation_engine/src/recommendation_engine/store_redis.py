"""Phase 4 — load enriched tracks + CLAP embeddings into Redis and build genre centroids.

Joins Artifact B (`tracks_enriched.json`) with Artifact C (`embeddings.jsonl`) by `id`,
creates the `idx:tracks` FLAT/COSINE vector index idempotently, writes one
`track:{deezer_id}` hash per track (metadata + float32 little-endian embedding bytes),
and computes/stores per-genre centroids at `genre_centroid:{genre_tag}`.

Product code uses redis-py directly (`FT.CREATE` / `FT.SEARCH` via `redis.Redis`).
The Redis MCP is only an inspection aid during development, never a runtime dependency.

Run:
    uv run python -m recommendation_engine.store_redis
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from . import config
from .contracts import EMBED_DIM, EnrichedTrack, load_embeddings, load_enriched_tracks

# float32 little-endian; vectors are binary, so the client must run in bytes mode.
EMBED_DTYPE = np.dtype("<f4")
EMBED_BYTES = EMBED_DIM * 4  # 512 * 4 = 2048

# Metadata fields persisted on each track hash (everything except `embedding`).
TRACK_METADATA_FIELDS = (
    "title",
    "artist",
    "album",
    "genre",
    "genre_tag",
    "isrc",
    "deezer_id",
    "spotify_id",
    "artwork_url",
    "duration_seconds",
    "rank",
)


# --- Connection --------------------------------------------------------------
def get_redis_client() -> redis.Redis:
    """Connect in bytes mode (`decode_responses=False`) using env via config helpers.

    Env vars (defaults: localhost:6379, no auth):
      REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD
    """
    config.load_dotenv()
    host = config.getenv("REDIS_HOST", "localhost")
    port = int(config.getenv("REDIS_PORT", "6379"))
    username = config.getenv("REDIS_USERNAME") or None
    password = config.getenv("REDIS_PASSWORD") or None
    return redis.Redis(
        host=host,
        port=port,
        username=username,
        password=password,
        decode_responses=False,
    )


# --- Index -------------------------------------------------------------------
def create_index(client: redis.Redis) -> bool:
    """Create `idx:tracks` if absent. Idempotent: returns False if it already exists."""
    schema = (
        TagField("genre_tag"),
        TagField("artist"),
        TextField("title"),
        VectorField(
            "embedding",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": EMBED_DIM,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )
    definition = IndexDefinition(
        prefix=[config.TRACK_KEY_PREFIX],
        index_type=IndexType.HASH,
    )
    try:
        client.ft(config.REDIS_INDEX).create_index(schema, definition=definition)
        return True
    except redis.ResponseError as exc:
        if "index already exists" in str(exc).lower():
            return False
        raise


# --- Encoding helpers --------------------------------------------------------
def encode_vector(vector: list[float]) -> bytes:
    """Encode a 512-float vector as float32 little-endian bytes (length 2048)."""
    arr = np.asarray(vector, dtype=EMBED_DTYPE)
    if arr.shape != (EMBED_DIM,):
        raise ValueError(f"expected {EMBED_DIM}-d vector, got shape {arr.shape}")
    blob = arr.tobytes()
    assert len(blob) == EMBED_BYTES, f"embedding must be {EMBED_BYTES} bytes, got {len(blob)}"
    return blob


def _track_hash_mapping(track: EnrichedTrack, vector: list[float]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    record = track.to_dict()
    for field in TRACK_METADATA_FIELDS:
        # redis-py encodes str/int/float; everything stored as UTF-8 bytes on read.
        mapping[field] = record[field]
    mapping["embedding"] = encode_vector(vector)
    return mapping


# --- Centroids ---------------------------------------------------------------
def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def store_track(client: redis.Redis, track: EnrichedTrack, vector: list[float]) -> None:
    """HSET a single track hash (idempotent overwrite)."""
    key = f"{config.TRACK_KEY_PREFIX}{track.deezer_id}"
    client.hset(key, mapping=_track_hash_mapping(track, vector))


def store_centroids(
    client: redis.Redis, vectors_by_genre: dict[str, list[list[float]]]
) -> int:
    """Compute and HSET one L2-normalized mean centroid per genre_tag."""
    for genre_tag, vectors in vectors_by_genre.items():
        mean = np.mean(np.asarray(vectors, dtype=EMBED_DTYPE), axis=0)
        centroid = _l2_normalize(mean).astype(EMBED_DTYPE)
        blob = centroid.tobytes()
        assert len(blob) == EMBED_BYTES, f"centroid must be {EMBED_BYTES} bytes, got {len(blob)}"
        key = f"{config.CENTROID_KEY_PREFIX}{genre_tag}"
        client.hset(
            key,
            mapping={
                "genre_tag": genre_tag,
                "count": len(vectors),
                "embedding": blob,
            },
        )
    return len(vectors_by_genre)


# --- Orchestration -----------------------------------------------------------
def load_into_redis(
    client: redis.Redis,
    enriched_path: Path | None = None,
    embeddings_path: Path | None = None,
) -> dict[str, int]:
    """Join B+C by id, create the index, store tracks, and build centroids.

    Returns a summary dict: {tracks, genres, centroids}.
    Idempotent — re-running overwrites the same keys cleanly.
    """
    enriched_path = enriched_path or config.TRACKS_ENRICHED_PATH
    embeddings_path = embeddings_path or config.EMBEDDINGS_PATH

    tracks = load_enriched_tracks(enriched_path)
    embeddings = load_embeddings(embeddings_path)
    vectors_by_id = {e["id"]: e["vector"] for e in embeddings}

    create_index(client)

    vectors_by_genre: dict[str, list[list[float]]] = defaultdict(list)
    loaded = 0
    for track in tracks:
        vector = vectors_by_id.get(track.id)
        if vector is None:
            # No embedding for this track — skip (cannot index without a vector).
            continue
        store_track(client, track, vector)
        vectors_by_genre[track.genre_tag].append(vector)
        loaded += 1

    centroids = store_centroids(client, vectors_by_genre)
    return {"tracks": loaded, "genres": len(vectors_by_genre), "centroids": centroids}


def main() -> None:
    client = get_redis_client()
    summary = load_into_redis(client)
    print(
        f"store_redis: loaded {summary['tracks']} tracks across "
        f"{summary['genres']} genres; wrote {summary['centroids']} centroids "
        f"into index {config.REDIS_INDEX!r}."
    )


if __name__ == "__main__":
    main()
