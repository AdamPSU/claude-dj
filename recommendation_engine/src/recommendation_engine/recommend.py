"""Phase 5 — recommendation engine.

Public surface:

    next_five(current_track_id, signal, recently_played=None) -> list[str]

Behavior (per IMPLEMENTATION_PLAN.md §9):
  1. Load the current track's embedding (float32 bytes) and ``genre_tag`` from Redis.
  2. Resolve the target genre:
       - signal in {"positive", "neutral"} -> the current ``genre_tag``.
       - signal == "negative" -> the genre whose centroid is MOST DISTANT (largest
         cosine distance) from the current vector, excluding the current genre.
         If only one genre exists, fall back to the same genre and log a warning.
  3. TAG-filtered KNN over ``idx:tracks`` (DIALECT 2, query vector as float32 bytes in
     PARAMS), over-fetching K > 5 so exclusions can be dropped and still return 5.
  4. Exclude ``current_track_id`` + ``recently_played``; rank ascending cosine distance.
  5. Deterministic tie-break by (score, rank, id); return up to 5 ids best-first.

This module only READS from Redis. Population is the job of Phase 4 / the test harness.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import redis

from . import config

logger = logging.getLogger(__name__)

# Buffer added on top of (5 + exclusions) so a tag-filtered KNN still yields 5
# distinct results after dropping the current track and recently-played ids.
_KNN_BUFFER = 10


# --- Redis connection --------------------------------------------------------
def get_redis_client() -> redis.Redis:
    """Bytes-mode client (vectors are binary). Reuses the project's env vars."""
    config.load_dotenv()
    host = config.getenv("REDIS_HOST", "localhost")
    port = int(config.getenv("REDIS_PORT", "6379") or "6379")
    username = config.getenv("REDIS_USERNAME") or None
    password = config.getenv("REDIS_PASSWORD")
    kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "username": username,
        "password": password or None,
        "decode_responses": False,
        "socket_timeout": float(config.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "10") or "10"),
        "socket_connect_timeout": float(config.getenv("REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS", "10") or "10"),
        "protocol": 3,
    }
    try:
        from redis.maint_notifications import MaintNotificationsConfig
    except ImportError:
        pass
    else:
        kwargs["maint_notifications_config"] = MaintNotificationsConfig(enabled=False)
    return redis.Redis(**kwargs)


# --- Low-level helpers -------------------------------------------------------
def _id_to_track_key(track_id: str) -> str:
    """'deezer:3135556' -> 'track:3135556'; a bare deezer id also works."""
    deezer_id = track_id.split(":", 1)[1] if track_id.startswith("deezer:") else track_id
    return f"{config.TRACK_KEY_PREFIX}{deezer_id}"


def _key_to_id(track_key: str) -> str:
    """'track:3135556' -> 'deezer:3135556'."""
    deezer_id = track_key.split(":", 1)[1] if ":" in track_key else track_key
    return f"deezer:{deezer_id}"


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _bytes_to_vec(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32)


def _load_track(client: redis.Redis, track_id: str) -> tuple[np.ndarray, str]:
    """Return (embedding vector, genre_tag) for a track id, or raise KeyError."""
    key = _id_to_track_key(track_id)
    fields = client.hmget(key, "embedding", "genre_tag")
    raw_vec, raw_genre = fields[0], fields[1]
    if raw_vec is None or raw_genre is None:
        raise KeyError(f"track not found or missing fields in Redis: {key}")
    return _bytes_to_vec(raw_vec), _decode(raw_genre)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity. Vectors are stored L2-normalized but normalize defensively."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def _resolve_target_genre(
    client: redis.Redis, current_vec: np.ndarray, current_genre: str, signal: str
) -> str:
    """Positive/neutral -> same genre. Negative -> most-distant centroid's genre."""
    if signal != "negative":
        return current_genre

    centroid_keys = [
        _decode(k) for k in client.scan_iter(match=f"{config.CENTROID_KEY_PREFIX}*")
    ]
    best_genre: str | None = None
    best_dist = -1.0
    for ckey in centroid_keys:
        genre_tag = ckey.split(":", 1)[1] if ":" in ckey else ckey
        if genre_tag == current_genre:
            continue
        raw = client.hget(ckey, "embedding")
        if raw is None:
            continue
        dist = _cosine_distance(current_vec, _bytes_to_vec(raw))
        # Deterministic tie-break across equally-distant genres: pick the smaller tag.
        if dist > best_dist or (dist == best_dist and (best_genre is None or genre_tag < best_genre)):
            best_dist = dist
            best_genre = genre_tag

    if best_genre is None:
        logger.warning(
            "negative feedback but no alternative genre centroid found; "
            "falling back to same genre %r",
            current_genre,
        )
        return current_genre
    return best_genre


def _knn(
    client: redis.Redis, genre_tag: str, query_vec: np.ndarray, k: int
) -> list[tuple[str, float, int]]:
    """Run a tag-filtered KNN. Returns [(id, score, rank), ...] ascending by score.

    Uses the raw ``FT.SEARCH`` command (not the redis-py Search helper) so parsing is
    stable across redis-py / RESP2 / RESP3 versions: the high-level ``.ft().search()``
    helper silently returns no docs against RESP3 dict responses on some versions.
    """
    query_str = f"(@genre_tag:{{{genre_tag}}})=>[KNN {k} @embedding $vec AS score]"
    raw = client.execute_command(
        "FT.SEARCH",
        config.REDIS_INDEX,
        query_str,
        "PARAMS", "2", "vec", query_vec.astype(np.float32).tobytes(),
        "SORTBY", "score", "ASC",
        "RETURN", "2", "score", "rank",
        "LIMIT", "0", str(k),
        "DIALECT", "2",
    )
    out: list[tuple[str, float, int]] = []
    for doc_id, fields in _iter_search_results(raw):
        track_id = _key_to_id(doc_id)
        field_map = _fields_to_map(fields)
        try:
            score = float(field_map.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        try:
            rank = int(field_map.get("rank", 0))
        except (TypeError, ValueError):
            rank = 0
        out.append((track_id, score, rank))
    return out


def _fields_to_map(fields: Any) -> dict[str, str]:
    """Normalize a field collection (flat [k,v,...] list or RESP3 dict) to a str map."""
    if isinstance(fields, dict):
        return {_decode(k): _decode(v) for k, v in fields.items()}
    out: dict[str, str] = {}
    seq = list(fields or [])
    for i in range(0, len(seq) - 1, 2):
        out[_decode(seq[i])] = _decode(seq[i + 1])
    return out


def _iter_search_results(raw: Any):
    """Yield (doc_id, fields) for both RESP2 (flat list) and RESP3 (dict) replies."""
    if isinstance(raw, dict):
        # RESP3: {"results": [{"id": ..., "extra_attributes"/"values": {...}}, ...]}
        for entry in raw.get(b"results", raw.get("results", [])) or []:
            doc_id = _decode(entry.get(b"id", entry.get("id")))
            fields = entry.get(b"extra_attributes", entry.get("extra_attributes"))
            if fields is None:
                fields = entry.get(b"values", entry.get("values"))
            yield doc_id, fields
        return
    # RESP2: [total, id1, [k,v,...], id2, [k,v,...], ...]
    if not raw:
        return
    i = 1
    while i < len(raw):
        doc_id = _decode(raw[i])
        fields = raw[i + 1] if i + 1 < len(raw) else []
        yield doc_id, fields
        i += 2


# --- Public API --------------------------------------------------------------
def next_five(
    current_track_id: str,
    signal: str,
    recently_played: list[str] | None = None,
    *,
    client: Any | None = None,
) -> list[str]:
    """Return up to 5 recommended track ids (best/most-similar first).

    See module docstring for the full algorithm. ``signal`` is one of
    "positive", "neutral", "negative".
    """
    if signal not in {"positive", "neutral", "negative"}:
        raise ValueError(f"signal must be positive|neutral|negative, got {signal!r}")

    exclusions = {current_track_id}
    if recently_played:
        exclusions.update(recently_played)

    client = client or get_redis_client()

    current_vec, current_genre = _load_track(client, current_track_id)
    target_genre = _resolve_target_genre(client, current_vec, current_genre, signal)

    k = 5 + len(exclusions) + _KNN_BUFFER
    candidates = _knn(client, target_genre, current_vec, k)

    # Stable sort by (score, rank, id) for deterministic, similarity-ascending order.
    candidates.sort(key=lambda c: (c[1], c[2], c[0]))

    results: list[str] = []
    for track_id, _score, _rank in candidates:
        if track_id in exclusions:
            continue
        results.append(track_id)
        if len(results) == 5:
            break
    return results
