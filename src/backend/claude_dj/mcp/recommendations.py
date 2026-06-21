from __future__ import annotations

import asyncio
import os
import socket
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RecommendedTrack:
    id: str
    title: str
    artist: str
    spotify_uri: str
    cluster: str
    duration_ms: int
    artwork_url: str = ""
    score: float | None = None
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "spotify_uri": self.spotify_uri,
            "cluster": self.cluster,
            "duration_ms": self.duration_ms,
            "artwork_url": self.artwork_url,
        }
        if self.score is not None:
            payload["score"] = self.score
        if self.rank is not None:
            payload["rank"] = self.rank
        return payload


@dataclass(frozen=True)
class RecommendationResult:
    available: bool
    source: str
    seed_track_id: str | None
    signal: str
    mode: str
    target_genre: str | None
    candidates: list[RecommendedTrack]
    reason: str | None = None
    dropped_hydration_count: int = 0


class RecommendationBackend(Protocol):
    async def recommend(
        self,
        *,
        seed_track_id: str,
        signal: str,
        mode: str,
        limit: int,
        avoid_clusters: list[str],
        exclude_track_ids: list[str],
    ) -> RecommendationResult: ...

    async def seed_candidates(self, *, limit: int, avoid_clusters: list[str]) -> list[RecommendedTrack]: ...


@dataclass(frozen=True)
class RedisRecommendationConfig:
    host: str = "localhost"
    port: int = 6379
    username: str | None = None
    password: str | None = None
    index: str = "idx:tracks"
    track_prefix: str = "track:"
    centroid_prefix: str = "genre_centroid:"
    socket_timeout_seconds: float = 10.0
    socket_connect_timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> RedisRecommendationConfig:
        return cls(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            username=os.environ.get("REDIS_USERNAME") or None,
            password=os.environ.get("REDIS_PASSWORD") or None,
            socket_timeout_seconds=float(os.environ.get("REDIS_SOCKET_TIMEOUT_SECONDS", "10")),
            socket_connect_timeout_seconds=float(os.environ.get("REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS", "10")),
        )

    def client_kwargs(self) -> dict[str, Any]:
        from redis.maint_notifications import MaintNotificationsConfig

        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": self.password,
            "decode_responses": False,
            "socket_timeout": self.socket_timeout_seconds,
            "socket_connect_timeout": self.socket_connect_timeout_seconds,
            "protocol": 3,
            "maint_notifications_config": MaintNotificationsConfig(enabled=False),
        }


class RedisRecommendationClient:
    def __init__(self, config: RedisRecommendationConfig | None = None, client: Any | None = None) -> None:
        self.config = config or RedisRecommendationConfig.from_env()
        self.client = client if client is not None else self._build_client()

    async def seed_candidates(self, *, limit: int, avoid_clusters: list[str]) -> list[RecommendedTrack]:
        return await asyncio.to_thread(self._seed_candidates_sync, limit, avoid_clusters)

    async def recommend(
        self,
        *,
        seed_track_id: str,
        signal: str,
        mode: str,
        limit: int,
        avoid_clusters: list[str],
        exclude_track_ids: list[str],
    ) -> RecommendationResult:
        return await asyncio.to_thread(
            self._recommend_sync,
            seed_track_id,
            signal,
            mode,
            limit,
            avoid_clusters,
            exclude_track_ids,
        )

    def _build_client(self) -> Any:
        return RawRedisClient(self.config)

    def _seed_candidates_sync(self, limit: int, avoid_clusters: list[str]) -> list[RecommendedTrack]:
        raw = self.client.execute_command(
            "FT.SEARCH",
            self.config.index,
            "*",
            "RETURN",
            "8",
            "title",
            "artist",
            "genre_tag",
            "spotify_id",
            "duration_seconds",
            "artwork_url",
            "rank",
            "deezer_id",
            "LIMIT",
            "0",
            str(max(1, limit) + len(avoid_clusters) + 20),
            "DIALECT",
            "2",
        )
        avoid = set(avoid_clusters)
        tracks: list[RecommendedTrack] = []
        for doc_id, fields in self._iter_search_results(raw):
            track_hash = self._fields_to_map(fields)
            if track_hash.get("genre_tag") in avoid:
                continue
            track = self._track_from_hash(
                self._key_to_id(doc_id),
                track_hash,
                score=0.0,
                rank=self._int_or_none(track_hash.get("rank")),
            )
            if track is not None:
                tracks.append(track)
        tracks.sort(key=lambda track: (-(track.rank or 0), track.id))
        return tracks[: max(1, limit)]

    def _recommend_sync(
        self,
        seed_track_id: str,
        signal: str,
        mode: str,
        limit: int,
        avoid_clusters: list[str],
        exclude_track_ids: list[str],
    ) -> RecommendationResult:
        seed_hash = self._track_hash(seed_track_id)
        seed_embedding = seed_hash.get("embedding")
        seed_genre = self._decode(seed_hash.get("genre_tag"))
        if seed_embedding is None or not seed_genre:
            return RecommendationResult(
                available=False,
                source="redis_vector",
                seed_track_id=seed_track_id,
                signal=signal,
                mode=mode,
                target_genre=None,
                candidates=[],
                reason="seed_track_missing_embedding_or_genre",
            )

        target_genre = seed_genre
        if signal == "negative" or mode in {"negative", "shift"}:
            target_genre = self._most_distant_genre(seed_embedding, seed_genre, set(avoid_clusters))

        if target_genre in set(avoid_clusters):
            return RecommendationResult(
                available=False,
                source="redis_vector",
                seed_track_id=seed_track_id,
                signal=signal,
                mode=mode,
                target_genre=target_genre,
                candidates=[],
                reason="target_genre_avoided",
            )

        exclusions = {seed_track_id, *exclude_track_ids}
        rows = self._knn(target_genre, seed_embedding, min(max(1, limit) + len(exclusions) + 2, 10))
        candidates: list[RecommendedTrack] = []
        dropped = 0
        for track_id, track_hash in rows:
            if track_id in exclusions:
                continue
            if self._decode(track_hash.get("genre_tag")) in set(avoid_clusters):
                continue
            track = self._track_from_hash(
                track_id,
                track_hash,
                score=self._float_or_zero(track_hash.get("score")),
                rank=self._int_or_none(track_hash.get("rank")),
            )
            if track is None:
                dropped += 1
                continue
            candidates.append(track)
            if len(candidates) >= max(1, limit):
                break
        return RecommendationResult(
            available=bool(candidates),
            source="redis_vector",
            seed_track_id=seed_track_id,
            signal=signal,
            mode=mode,
            target_genre=target_genre,
            candidates=candidates,
            reason=None if candidates else "no_playable_candidates",
            dropped_hydration_count=dropped,
        )

    def _knn(self, genre_tag: str, query_vec: bytes, k: int) -> list[tuple[str, dict[str, str]]]:
        raw = self.client.execute_command(
            "FT.SEARCH",
            self.config.index,
            f"(@genre_tag:{{{genre_tag}}})=>[KNN {k} @embedding $vec AS score]",
            "PARAMS",
            "2",
            "vec",
            query_vec,
            "SORTBY",
            "score",
            "ASC",
            "RETURN",
            "8",
            "score",
            "rank",
            "title",
            "artist",
            "genre_tag",
            "spotify_id",
            "duration_seconds",
            "artwork_url",
            "LIMIT",
            "0",
            str(k),
            "DIALECT",
            "2",
        )
        rows: list[tuple[str, dict[str, str]]] = []
        for doc_id, fields in self._iter_search_results(raw):
            field_map = self._fields_to_map(fields)
            rows.append((self._key_to_id(doc_id), field_map))
        rows.sort(
            key=lambda row: (
                self._float_or_zero(row[1].get("score")),
                self._int_or_none(row[1].get("rank")) or 0,
                row[0],
            )
        )
        return rows

    def _track_hash(self, track_id: str) -> dict[str, Any]:
        raw = self.client.hgetall(self._id_to_track_key(track_id))
        return {self._decode(key): value for key, value in raw.items()}

    def _track_from_hash(
        self,
        track_id: str,
        track_hash: dict[str, Any],
        *,
        score: float | None,
        rank: int | None,
    ) -> RecommendedTrack | None:
        title = self._decode(track_hash.get("title"))
        artist = self._decode(track_hash.get("artist"))
        genre = self._decode(track_hash.get("genre_tag"))
        spotify_id = self._decode(track_hash.get("spotify_id"))
        duration_seconds = self._decode(track_hash.get("duration_seconds"))
        if not title or not artist or not genre or not spotify_id or not duration_seconds:
            return None
        try:
            duration_ms = int(duration_seconds) * 1000
        except ValueError:
            return None
        spotify_uri = spotify_id if spotify_id.startswith("spotify:") else f"spotify:track:{spotify_id}"
        return RecommendedTrack(
            id=track_id,
            title=title,
            artist=artist,
            spotify_uri=spotify_uri,
            cluster=genre,
            duration_ms=duration_ms,
            artwork_url=self._decode(track_hash.get("artwork_url")),
            score=score,
            rank=rank if rank is not None else self._int_or_none(self._decode(track_hash.get("rank"))),
        )

    def _most_distant_genre(self, seed_embedding: bytes, seed_genre: str, avoid_clusters: set[str]) -> str:
        import numpy as np

        seed = np.frombuffer(seed_embedding, dtype=np.float32)
        best_genre: str | None = None
        best_distance = -1.0
        for key in self.client.execute_command("KEYS", f"{self.config.centroid_prefix}*"):
            key_text = self._decode(key)
            genre = key_text.split(":", 1)[1] if ":" in key_text else key_text
            if genre == seed_genre or genre in avoid_clusters:
                continue
            centroid = self.client.hget(key_text, "embedding")
            if centroid is None:
                continue
            distance = self._cosine_distance(seed, np.frombuffer(centroid, dtype=np.float32))
            if distance > best_distance or (distance == best_distance and (best_genre is None or genre < best_genre)):
                best_distance = distance
                best_genre = genre
        return best_genre or seed_genre

    def _id_to_track_key(self, track_id: str) -> str:
        deezer_id = track_id.split(":", 1)[1] if track_id.startswith("deezer:") else track_id
        return f"{self.config.track_prefix}{deezer_id}"

    @staticmethod
    def _key_to_id(track_key: Any) -> str:
        key = RedisRecommendationClient._decode(track_key)
        deezer_id = key.split(":", 1)[1] if ":" in key else key
        return f"deezer:{deezer_id}"

    @classmethod
    def _fields_to_map(cls, fields: Any) -> dict[str, str]:
        if isinstance(fields, dict):
            return {cls._decode(key): cls._decode(value) for key, value in fields.items()}
        seq = list(fields or [])
        return {cls._decode(seq[index]): cls._decode(seq[index + 1]) for index in range(0, len(seq) - 1, 2)}

    @classmethod
    def _iter_search_results(cls, raw: Any):
        if isinstance(raw, dict):
            for entry in raw.get(b"results", raw.get("results", [])) or []:
                doc_id = entry.get(b"id", entry.get("id"))
                fields = entry.get(b"extra_attributes", entry.get("extra_attributes")) or entry.get(b"values", entry.get("values"))
                yield doc_id, fields
            return
        if not raw:
            return
        index = 1
        while index < len(raw):
            yield raw[index], raw[index + 1] if index + 1 < len(raw) else []
            index += 2

    @staticmethod
    def _cosine_distance(a, b) -> float:
        import numpy as np

        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 1.0
        return 1.0 - float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _decode(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _float_or_zero(value: str | None) -> float:
        try:
            return float(value or 0.0)
        except ValueError:
            return 0.0

    @staticmethod
    def _int_or_none(value: str | None) -> int | None:
        try:
            return int(value) if value not in {None, ""} else None
        except ValueError:
            return None


class RawRedisClient:
    def __init__(self, config: RedisRecommendationConfig) -> None:
        self.config = config

    def ping(self) -> bool:
        return self.execute_command("PING") == b"PONG"

    def hgetall(self, key: str) -> dict[bytes, bytes]:
        response = self.execute_command("HGETALL", key)
        if not response:
            return {}
        return {response[index]: response[index + 1] for index in range(0, len(response) - 1, 2)}

    def hget(self, key: str, field: str) -> bytes | None:
        return self.execute_command("HGET", key, field)

    def execute_command(self, *parts: Any) -> Any:
        last_error: OSError | TimeoutError | None = None
        for _ in range(3):
            try:
                return self._execute_command_once(*parts)
            except (OSError, TimeoutError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Redis command failed before execution")

    def _execute_command_once(self, *parts: Any) -> Any:
        with socket.create_connection(
            (self.config.host, self.config.port),
            timeout=self.config.socket_connect_timeout_seconds,
        ) as sock:
            sock.settimeout(self.config.socket_timeout_seconds)
            sock.sendall(self._command(*self._hello_parts()))
            self._read_response(sock)
            sock.sendall(self._command(*parts))
            return self._read_response(sock)

    def _hello_parts(self) -> tuple[Any, ...]:
        if self.config.username or self.config.password:
            return ("HELLO", "2", "AUTH", self.config.username or "default", self.config.password or "")
        return ("HELLO", "2")

    def _command(self, *parts: Any) -> bytes:
        return b"*" + str(len(parts)).encode() + b"\r\n" + b"".join(self._bulk(part) for part in parts)

    def _bulk(self, value: Any) -> bytes:
        if isinstance(value, bytes):
            data = value
        else:
            data = str(value).encode()
        return b"$" + str(len(data)).encode() + b"\r\n" + data + b"\r\n"

    def _read_response(self, sock: socket.socket) -> Any:
        line = self._read_line(sock)
        prefix = line[:1]
        body = line[1:-2]
        if prefix == b"+":
            return body
        if prefix == b":":
            return int(body)
        if prefix == b"-":
            raise RuntimeError(body.decode(errors="replace"))
        if prefix == b"$":
            length = int(body)
            if length < 0:
                return None
            payload = self._read_exact(sock, length + 2)
            return payload[:-2]
        if prefix == b"*":
            return [self._read_response(sock) for _ in range(int(body))]
        raise RuntimeError(f"unsupported Redis response prefix: {prefix!r}")

    def _read_line(self, sock: socket.socket) -> bytes:
        line = b""
        while not line.endswith(b"\r\n"):
            chunk = sock.recv(1)
            if not chunk:
                raise RuntimeError("Redis socket closed")
            line += chunk
        return line

    def _read_exact(self, sock: socket.socket, length: int) -> bytes:
        data = b""
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                raise RuntimeError("Redis socket closed")
            data += chunk
        return data
