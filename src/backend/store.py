"""Redis data layer for ClaudeDJ.

Handles reaction traces (FR-17), session state (FR-18), memory (FR-19),
and the compact session context bundle (FR-20). Also provides track
profile storage and vector search index management for the embedding
pipeline.

Key conventions:
    track:{spotify_id}          — JSON track profile + embedding
    session:current             — JSON current session state
    session:current:queue       — list of queued track IDs
    session:current:recent      — list of recently played track IDs
    memory:liked_clusters       — set of liked cluster names
    memory:disliked_clusters    — set of disliked cluster names
    memory:skip_history         — list of recently skipped track IDs
    reaction:{track_id}         — JSON reaction trace for a track
    stream:reactions            — stream of reaction events
    stream:playback             — stream of playback events
    idx:tracks                  — RediSearch vector index
"""

from __future__ import annotations

import json
import time

import redis

from reaction import ReactionFrame, ReactionScore, Sentiment, TrackReaction


def connect(url: str = "redis://localhost:6379/0") -> redis.Redis:
    """Create a Redis connection."""
    return redis.Redis.from_url(url, decode_responses=True)


# ---------------------------------------------------------------------------
# Track profiles
# ---------------------------------------------------------------------------


def store_track(r: redis.Redis, track: dict) -> None:
    """Store a track profile as JSON.

    Expected keys: id, title, artist, album, spotify_uri, artwork_url,
    cluster, energy, has_lyrics, embedding (list[float]).
    """
    track_id = track["id"]
    r.json().set(f"track:{track_id}", "$", track)


def get_track(r: redis.Redis, track_id: str) -> dict | None:
    """Retrieve a track profile."""
    data = r.json().get(f"track:{track_id}")
    return data if data else None


# ---------------------------------------------------------------------------
# Reaction traces (FR-17)
# ---------------------------------------------------------------------------


def store_reaction_frame(r: redis.Redis, track_id: str, frame: ReactionFrame) -> None:
    """Append a reaction frame to a track's trace and to the reaction stream."""
    frame_data = {
        "timestamp": frame.timestamp,
        "presence": frame.presence,
        "movement": frame.movement,
        "face": frame.face,
        "playback": frame.playback,
        "vocal": frame.vocal,
        "source": frame.source.value,
    }

    # Append to track-specific list
    r.rpush(f"reaction:{track_id}:frames", json.dumps(frame_data))

    # Write to reaction stream
    stream_data = {"track_id": track_id, **{k: str(v) for k, v in frame_data.items()}}
    r.xadd("stream:reactions", stream_data, maxlen=1000)


def store_reaction_score(
    r: redis.Redis, track_id: str, score: ReactionScore
) -> None:
    """Store a windowed reaction score for a track."""
    score_data = {
        "score": score.score,
        "confidence": score.confidence,
        "sentiment": score.sentiment.value,
        "window_start": score.window_start,
        "window_end": score.window_end,
        "frame_count": score.frame_count,
        "source": score.source.value,
    }
    r.rpush(f"reaction:{track_id}:scores", json.dumps(score_data))


def get_track_reaction(r: redis.Redis, track_id: str) -> dict:
    """Get full reaction trace for a track."""
    frames_raw = r.lrange(f"reaction:{track_id}:frames", 0, -1)
    scores_raw = r.lrange(f"reaction:{track_id}:scores", 0, -1)
    return {
        "track_id": track_id,
        "frames": [json.loads(f) for f in frames_raw],
        "scores": [json.loads(s) for s in scores_raw],
    }


# ---------------------------------------------------------------------------
# Session state (FR-18)
# ---------------------------------------------------------------------------

_DEFAULT_SESSION = {
    "session_id": "",
    "current_track": None,
    "current_cluster": None,
    "cluster_streak": 0,
    "min_cluster_run": 3,
    "max_cluster_run": 6,
    "dj_status": "listening",
    "started_at": 0.0,
}


def init_session(r: redis.Redis, session_id: str | None = None) -> dict:
    """Initialize a fresh session."""
    session = {
        **_DEFAULT_SESSION,
        "session_id": session_id or f"session-{int(time.time())}",
        "started_at": time.time(),
    }
    r.json().set("session:current", "$", session)
    r.delete("session:current:queue", "session:current:recent")
    return session


def get_session(r: redis.Redis) -> dict:
    """Get current session state."""
    data = r.json().get("session:current")
    if not data:
        return init_session(r)
    data["queue"] = r.lrange("session:current:queue", 0, -1)
    data["recent_tracks"] = r.lrange("session:current:recent", 0, -1)
    return data


def update_session(r: redis.Redis, **fields) -> None:
    """Update specific session fields."""
    for key, value in fields.items():
        r.json().set("session:current", f"$.{key}", value)


def set_current_track(r: redis.Redis, track_id: str, cluster: str | None = None) -> None:
    """Set the currently playing track and push it to recent."""
    update_session(r, current_track=track_id)
    if cluster:
        # Check if cluster changed
        session = r.json().get("session:current")
        old_cluster = session.get("current_cluster") if session else None
        if cluster == old_cluster:
            update_session(r, cluster_streak=session.get("cluster_streak", 0) + 1)
        else:
            update_session(r, current_cluster=cluster, cluster_streak=1)

    # Add to recent, keep last 20
    r.lpush("session:current:recent", track_id)
    r.ltrim("session:current:recent", 0, 19)

    # Log to playback stream
    r.xadd("stream:playback", {
        "event": "track_start",
        "track_id": track_id,
        "cluster": cluster or "",
        "timestamp": str(time.time()),
    }, maxlen=500)


def set_queue(r: redis.Redis, track_ids: list[str]) -> None:
    """Replace the current queue."""
    r.delete("session:current:queue")
    if track_ids:
        r.rpush("session:current:queue", *track_ids)


def set_dj_status(r: redis.Redis, status: str) -> None:
    """Update the DJ status line."""
    update_session(r, dj_status=status)


# ---------------------------------------------------------------------------
# Memory (FR-19)
# ---------------------------------------------------------------------------


def mark_feedback(
    r: redis.Redis,
    track_id: str,
    sentiment: str,
    cluster: str | None = None,
) -> None:
    """Record track feedback and update cluster memory."""
    r.xadd("stream:reactions", {
        "event": "feedback",
        "track_id": track_id,
        "sentiment": sentiment,
        "cluster": cluster or "",
        "timestamp": str(time.time()),
    }, maxlen=1000)

    if cluster:
        if sentiment == "positive":
            r.sadd("memory:liked_clusters", cluster)
        elif sentiment == "negative":
            r.sadd("memory:disliked_clusters", cluster)
            r.srem("memory:liked_clusters", cluster)


def get_memory(r: redis.Redis) -> dict:
    """Get the current memory state."""
    return {
        "liked_clusters": list(r.smembers("memory:liked_clusters")),
        "disliked_clusters": list(r.smembers("memory:disliked_clusters")),
        "recent_skips": r.lrange("memory:skip_history", 0, 9),
    }


def record_skip(r: redis.Redis, track_id: str) -> None:
    """Record a skipped track."""
    r.lpush("memory:skip_history", track_id)
    r.ltrim("memory:skip_history", 0, 19)


# ---------------------------------------------------------------------------
# Session context bundle (FR-20)
# ---------------------------------------------------------------------------


def get_session_context(r: redis.Redis, reaction_summary: dict | None = None) -> dict:
    """Build the compact decision bundle for the agent.

    This is what get_session_context MCP tool returns. Contains everything
    Claude needs to make the next queue decision, nothing more.
    """
    session = get_session(r)
    memory = get_memory(r)

    # Get current track details if available
    current_track_data = None
    if session.get("current_track"):
        current_track_data = get_track(r, session["current_track"])

    context = {
        "current_track": current_track_data,
        "queue": session.get("queue", []),
        "recent_tracks": session.get("recent_tracks", []),
        "current_cluster": session.get("current_cluster"),
        "cluster_streak": session.get("cluster_streak", 0),
        "cluster_policy": {
            "min": session.get("min_cluster_run", 3),
            "max": session.get("max_cluster_run", 6),
        },
        "dj_status": session.get("dj_status", "listening"),
        "liked_clusters": memory["liked_clusters"],
        "disliked_clusters": memory["disliked_clusters"],
        "recent_skips": memory["recent_skips"],
    }

    if reaction_summary:
        context["reaction"] = reaction_summary
    else:
        context["reaction"] = {
            "current_score": 0.5,
            "confidence": 0.0,
            "sentiment": "neutral",
            "trend_direction": "stable",
        }

    # Compute recommended action
    streak = context["cluster_streak"]
    policy = context["cluster_policy"]
    sentiment = context["reaction"]["sentiment"]

    if sentiment == "negative" and streak >= 1:
        context["recommended_action"] = "shift_away"
    elif streak >= policy["max"]:
        context["recommended_action"] = "shift_adjacent"
    elif sentiment == "positive" and streak < policy["max"]:
        context["recommended_action"] = "stay_close"
    elif sentiment == "neutral" and streak >= policy["min"]:
        context["recommended_action"] = "prepare_slight_shift"
    else:
        context["recommended_action"] = "hold"

    return context


# ---------------------------------------------------------------------------
# Vector index (for embedding pipeline)
# ---------------------------------------------------------------------------


def create_track_index(r: redis.Redis, dim: int = 1536) -> None:
    """Create the RediSearch vector index for track embeddings.

    Call once during setup. Uses HNSW for approximate nearest neighbor.
    """
    try:
        r.ft("idx:tracks").info()
        return  # index already exists
    except redis.ResponseError:
        pass

    r.ft("idx:tracks").create_index(
        fields=[
            redis.commands.search.field.TextField("title"),
            redis.commands.search.field.TextField("artist"),
            redis.commands.search.field.TagField("cluster"),
            redis.commands.search.field.NumericField("energy"),
            redis.commands.search.field.VectorField(
                "embedding",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
            ),
        ],
        definition=redis.commands.search.indexDefinition.IndexDefinition(
            prefix=["track:"],
            index_type=redis.commands.search.indexDefinition.IndexType.JSON,
        ),
    )
