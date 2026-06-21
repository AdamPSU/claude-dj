from __future__ import annotations

import os
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .recommendations import RecommendationBackend, RecommendedTrack


@dataclass(frozen=True)
class Track:
    id: str
    title: str
    artist: str
    spotify_uri: str
    cluster: str
    duration_ms: int = 180_000
    artwork_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "spotify_uri": self.spotify_uri,
            "cluster": self.cluster,
            "duration_ms": self.duration_ms,
            "artwork_url": self.artwork_url,
        }


@dataclass(frozen=True)
class SpotifyDevice:
    id: str | None
    name: str
    volume_percent: int | None = None
    type: str | None = None
    is_active: bool = False
    is_restricted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "volume_percent": self.volume_percent,
            "type": self.type,
            "is_active": self.is_active,
            "is_restricted": self.is_restricted,
        }


@dataclass(frozen=True)
class SpotifyPlaybackState:
    track_id: str | None
    spotify_uri: str | None
    progress_ms: int | None
    duration_ms: int | None
    is_playing: bool
    device: SpotifyDevice | None = None


@dataclass(frozen=True)
class SpotifyPlaylist:
    id: str
    name: str
    public: bool | None = None
    collaborative: bool = False
    total_tracks: int = 0


class SpotifyPlayer(Protocol):
    async def start_track(self, spotify_uri: str) -> None: ...

    async def set_playback_volume(self, volume_percent: int) -> None: ...

    async def pause_playback(self) -> None: ...

    async def resume_playback(self) -> None: ...

    async def get_current_playback(self) -> SpotifyPlaybackState | None: ...

    async def search_tracks(self, query: str, limit: int = 6) -> list[Track]: ...

    async def list_user_playlists(self, limit: int = 20) -> list[SpotifyPlaylist]: ...

    async def list_playlist_tracks(self, playlist_id: str, playlist_name: str, limit: int = 100) -> list[Track]: ...

    async def list_devices(self) -> list[SpotifyDevice]: ...

    async def transfer_playback(self, device_id: str, *, play: bool = False) -> None: ...


class NoopSpotifyPlayer:
    async def start_track(self, spotify_uri: str) -> None:
        return None

    async def set_playback_volume(self, volume_percent: int) -> None:
        return None

    async def pause_playback(self) -> None:
        return None

    async def resume_playback(self) -> None:
        return None

    async def get_current_playback(self) -> SpotifyPlaybackState | None:
        return None

    async def search_tracks(self, query: str, limit: int = 6) -> list[Track]:
        return []

    async def list_user_playlists(self, limit: int = 20) -> list[SpotifyPlaylist]:
        return []

    async def list_playlist_tracks(self, playlist_id: str, playlist_name: str, limit: int = 100) -> list[Track]:
        return []

    async def list_devices(self) -> list[SpotifyDevice]:
        return []

    async def transfer_playback(self, device_id: str, *, play: bool = False) -> None:
        return None


class InMemoryPlaybackRuntime:
    def __init__(
        self,
        *,
        tracks: list[Track] | None = None,
        spotify: SpotifyPlayer | None = None,
        recommendations: RecommendationBackend | None = None,
        seed_vibe: str = "playlist-informed autonomous start",
        initial_seed_track_id: str | None = None,
        require_recommendations: bool = False,
        playlist_limit: int = 5,
        playlist_track_limit: int = 50,
        demo_track_seconds: int | None = None,
        queue_min_tracks: int | None = None,
        queue_max_tracks: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.tracks = {track.id: track for track in (tracks if tracks is not None else default_demo_tracks())}
        self.spotify = spotify or NoopSpotifyPlayer()
        self.recommendations = recommendations
        self.seed_vibe = seed_vibe
        self.initial_seed_track_id = initial_seed_track_id
        self.require_recommendations = require_recommendations
        self.current_track_id: str | None = None
        self.queue_track_ids: list[str] = []
        self.pending_queue_track_ids: list[str] = []
        self.recent_track_ids: list[str] = []
        self.current_cluster: str | None = None
        self.cluster_streak = 0
        self._spotify_playlist_catalog_loaded = False
        self.playlist_limit = playlist_limit
        self.playlist_track_limit = playlist_track_limit
        self.demo_track_seconds = demo_track_seconds
        self.queue_min_tracks = queue_min_tracks
        self.queue_max_tracks = queue_max_tracks
        self.clock = clock
        self._current_track_started_at: float | None = None
        self._current_track_paused_at: float | None = None
        self._current_track_paused_seconds = 0.0
        self.preferred_device_id: str | None = None
        self._playlist_names_cache: list[str] | None = None

    async def search_track_embeddings(
        self,
        *,
        query: str | None = None,
        mode: str = "text",
        seed_track_id: str | None = None,
        signal: str | None = None,
        avoid_clusters: list[str] | None = None,
        exclude_recent: bool = False,
        exclude_track_ids: list[str] | None = None,
        limit: int = 6,
    ) -> dict[str, Any]:
        if self.recommendations is not None:
            seed = self._resolved_seed_track_id(seed_track_id)
            if seed is None:
                if self.require_recommendations:
                    raise ValueError("missing Redis seed track id")
                return self._unavailable_recommendation("missing_seed_track_id", query, mode, avoid_clusters)
            exclusions = list(exclude_track_ids or [])
            if exclude_recent:
                exclusions.extend([self.current_track_id] if self.current_track_id else [])
                exclusions.extend(self.recent_track_ids)
                exclusions.extend(self.queue_track_ids)
                exclusions.extend(self.pending_queue_track_ids)
            exclusions = list(dict.fromkeys(track_id for track_id in exclusions if track_id and track_id != seed))
            result = await self.recommendations.recommend(
                seed_track_id=seed,
                signal=signal or self._signal_from_mode(mode),
                mode=mode,
                limit=limit,
                avoid_clusters=avoid_clusters or [],
                exclude_track_ids=exclusions,
            )
            seed_track = await self._recommended_track_by_id(result.seed_track_id)
            if seed_track is not None:
                self._register_recommended_tracks([seed_track])
            self._register_recommended_tracks(result.candidates)
            if self.require_recommendations and not result.available:
                raise ValueError(result.reason or "Redis recommendations unavailable")
            return {
                "available": result.available,
                "stub": False,
                "source": result.source,
                "reason": result.reason,
                "query": query,
                "mode": mode,
                "seed_track_id": result.seed_track_id,
                "signal": result.signal,
                "target_genre": result.target_genre,
                "avoid_clusters": avoid_clusters or [],
                "exclude_track_ids": exclusions,
                "candidate_count": len(result.candidates),
                "dropped_hydration_count": result.dropped_hydration_count,
                "candidates": [track.to_dict() for track in result.candidates],
            }
        if self.require_recommendations:
            raise ValueError("Redis recommendations are required but not configured")
        avoid = set(avoid_clusters or [])
        spotify_candidates = await self._spotify_candidates(
            query=query or self.seed_vibe,
            mode=mode,
            avoid_clusters=avoid,
            limit=limit,
        )
        if spotify_candidates:
            return {
                "available": True,
                "stub": False,
                "source": "spotify_playlist_search",
                "temporary_until_embeddings": True,
                "query": query,
                "mode": mode,
                "avoid_clusters": avoid_clusters or [],
                "candidates": [
                    {**track.to_dict(), "score": round(max(0.01, 1.0 - (index * 0.03)), 2)}
                    for index, track in enumerate(spotify_candidates)
                ],
            }

        tracks = [track for track in self.tracks.values() if track.cluster not in avoid]
        if mode in {"shift", "adjacent_shift", "slight_shift"} and self.current_cluster:
            shifted = [track for track in tracks if track.cluster != self.current_cluster]
            tracks = shifted or tracks
        capped = tracks[: max(1, min(limit, 12))]
        return {
            "available": False,
            "stub": True,
            "source": "demo_catalog",
            "query": query,
            "mode": mode,
            "avoid_clusters": avoid_clusters or [],
            "candidates": [
                {**track.to_dict(), "score": round(1.0 - (index * 0.01), 2)}
                for index, track in enumerate(capped)
            ],
        }

    async def get_seed_candidates(self, *, limit: int = 12, avoid_clusters: list[str] | None = None) -> dict[str, Any]:
        if self.recommendations is None:
            if self.require_recommendations:
                raise ValueError("Redis recommendations are required but not configured")
            return {"available": False, "source": "redis_seed_candidates", "reason": "recommendation_backend_unavailable", "candidates": []}
        candidates = await self.recommendations.seed_candidates(limit=limit, avoid_clusters=avoid_clusters or [])
        self._register_recommended_tracks(candidates)
        return {
            "available": bool(candidates),
            "source": "redis_seed_candidates",
            "avoid_clusters": avoid_clusters or [],
            "candidates": [track.to_dict() for track in candidates],
        }

    async def replace_queue(self, track_ids: list[str], *, reason: str, timing: str = "now") -> dict[str, Any]:
        self._validate_track_ids(track_ids)
        capped_track_ids = self._queue_cap(track_ids)
        dropped_track_ids = track_ids[len(capped_track_ids):]
        if timing == "after_current_track":
            self.pending_queue_track_ids = list(capped_track_ids)
        else:
            self.queue_track_ids = list(capped_track_ids)
            self.pending_queue_track_ids = []
        return {
            "accepted": True,
            "stub": False,
            "source": "app_queue",
            "track_ids": list(capped_track_ids),
            "dropped_track_ids": list(dropped_track_ids),
            "queue_track_ids": list(self.queue_track_ids),
            "pending_queue_track_ids": list(self.pending_queue_track_ids),
            "reason": reason,
            "timing": timing,
        }

    async def play_track(self, track_id: str) -> dict[str, Any]:
        track = self._get_track(track_id)
        await self._ensure_spotify_device()
        await self.spotify.start_track(track.spotify_uri)
        if track_id in self.pending_queue_track_ids:
            self.queue_track_ids = list(self.pending_queue_track_ids)
            self.pending_queue_track_ids = []
        self.queue_track_ids = [queued_id for queued_id in self.queue_track_ids if queued_id != track_id]
        self._set_current_track(track)
        return {"started": True, "stub": False, "track_id": track.id, "spotify_uri": track.spotify_uri}

    async def play_next_queued_track(self) -> dict[str, Any] | None:
        if not self.queue_track_ids and self.pending_queue_track_ids:
            return await self.play_track(self.pending_queue_track_ids[0])
        if not self.queue_track_ids:
            return None
        return await self.play_track(self.queue_track_ids[0])

    async def pause_music(self) -> None:
        await self.spotify.pause_playback()
        if self._current_track_started_at is not None and self._current_track_paused_at is None:
            self._current_track_paused_at = self.clock()

    async def resume_music(self) -> None:
        await self.spotify.resume_playback()
        if self._current_track_paused_at is not None:
            self._current_track_paused_seconds += max(0.0, self.clock() - self._current_track_paused_at)
            self._current_track_paused_at = None

    async def get_music_volume(self) -> int:
        spotify_state = await self.spotify.get_current_playback()
        if spotify_state and spotify_state.device and spotify_state.device.volume_percent is not None:
            return self._volume_percent(spotify_state.device.volume_percent)
        return 100

    async def set_music_volume(self, volume_percent: int) -> None:
        await self.spotify.set_playback_volume(self._volume_percent(volume_percent))

    async def get_current_playback(self) -> dict[str, Any]:
        spotify_state = await self.spotify.get_current_playback()
        current_track = self._track_from_spotify_state(spotify_state) or self._current_track()
        if current_track and self.current_track_id is None:
            self._set_current_track(current_track)

        raw_progress_ms = spotify_state.progress_ms if spotify_state else 0
        raw_progress_ms = max(raw_progress_ms or 0, self._demo_wall_clock_progress_ms(current_track))
        raw_duration_ms = spotify_state.duration_ms if spotify_state and spotify_state.duration_ms else current_track.duration_ms if current_track else 0
        duration_ms = self._effective_duration_ms(raw_duration_ms)
        progress_ms = min(raw_progress_ms or 0, duration_ms or 0) if duration_ms else raw_progress_ms
        seconds_remaining = max(0, int(((duration_ms or 0) - (progress_ms or 0)) / 1000))
        return {
            "available": current_track is not None,
            "current_track_id": current_track.id if current_track else None,
            "current_track": current_track.to_dict() if current_track else None,
            "current_cluster": current_track.cluster if current_track else self.current_cluster,
            "is_playing": spotify_state.is_playing if spotify_state else False,
            "progress_ms": progress_ms,
            "duration_ms": duration_ms,
            "seconds_remaining": seconds_remaining,
            "queue_track_ids": list(self.queue_track_ids),
            "pending_queue_track_ids": list(self.pending_queue_track_ids),
            "recent_track_ids": list(self.recent_track_ids),
            "cluster_streak": self.cluster_streak,
            "device": spotify_state.device.to_dict() if spotify_state and spotify_state.device else None,
        }

    def _effective_duration_ms(self, duration_ms: int | None) -> int:
        if not duration_ms:
            return 0
        if self.demo_track_seconds is None:
            return duration_ms
        return min(duration_ms, max(1, self.demo_track_seconds) * 1000)

    def _demo_wall_clock_progress_ms(self, current_track: Track | None) -> int:
        if self.demo_track_seconds is None or current_track is None or self._current_track_started_at is None:
            return 0
        if current_track.id != self.current_track_id:
            return 0
        paused_seconds = self._current_track_paused_seconds
        if self._current_track_paused_at is not None:
            paused_seconds += max(0.0, self.clock() - self._current_track_paused_at)
        return max(0, int((self.clock() - self._current_track_started_at - paused_seconds) * 1000))

    async def get_session_context(self) -> dict[str, Any]:
        playback = await self.get_current_playback()
        playlist_names = await self._playlist_names()
        return {
            "seed_vibe": self.seed_vibe,
            "initial_seed_track_id": self.initial_seed_track_id,
            "redis_recommendations_required": self.require_recommendations,
            "available_playlist_names": playlist_names,
            "current_track_id": playback["current_track_id"],
            "current_track": playback["current_track"],
            "current_cluster": playback["current_cluster"],
            "queue_track_ids": playback["queue_track_ids"],
            "pending_queue_track_ids": playback["pending_queue_track_ids"],
            "recent_track_ids": playback["recent_track_ids"],
            "cluster_streak": self.cluster_streak,
            "seconds_remaining": playback["seconds_remaining"],
            "recommended_next_action": "start_initial_set" if playback["current_track_id"] is None else "keep_current_set_ready",
        }

    def _resolved_seed_track_id(self, seed_track_id: str | None) -> str | None:
        if seed_track_id:
            return seed_track_id
        if self.current_track_id and self.current_track_id.startswith("deezer:"):
            return self.current_track_id
        return self.initial_seed_track_id

    def _unavailable_recommendation(
        self,
        reason: str,
        query: str | None,
        mode: str,
        avoid_clusters: list[str] | None,
    ) -> dict[str, Any]:
        return {
            "available": False,
            "stub": False,
            "source": "redis_vector",
            "reason": reason,
            "query": query,
            "mode": mode,
            "avoid_clusters": avoid_clusters or [],
            "candidate_count": 0,
            "candidates": [],
        }

    @staticmethod
    def _signal_from_mode(mode: str) -> str:
        if mode in {"negative", "shift"}:
            return "negative"
        return "neutral"

    def _register_recommended_tracks(self, tracks: list[RecommendedTrack]) -> None:
        self._register_tracks(
            [
                Track(
                    id=track.id,
                    title=track.title,
                    artist=track.artist,
                    spotify_uri=track.spotify_uri,
                    cluster=track.cluster,
                    duration_ms=track.duration_ms,
                    artwork_url=track.artwork_url,
                )
                for track in tracks
            ]
        )

    async def _recommended_track_by_id(self, track_id: str | None) -> RecommendedTrack | None:
        if not track_id or self.recommendations is None or not hasattr(self.recommendations, "get_track"):
            return None
        return await self.recommendations.get_track(track_id)

    async def _playlist_names(self) -> list[str]:
        if self._playlist_names_cache is not None:
            return list(self._playlist_names_cache)
        playlists = await self.spotify.list_user_playlists(limit=self.playlist_limit)
        self._playlist_names_cache = [playlist.name for playlist in playlists if playlist.name]
        return list(self._playlist_names_cache)

    def _set_current_track(self, track: Track) -> None:
        previous_track_id = self.current_track_id
        if self.current_cluster == track.cluster:
            self.cluster_streak += 1
        else:
            self.current_cluster = track.cluster
            self.cluster_streak = 1
        self.current_track_id = track.id
        if previous_track_id != track.id:
            self._current_track_started_at = self.clock()
            self._current_track_paused_at = None
            self._current_track_paused_seconds = 0.0
        if track.id not in self.recent_track_ids:
            self.recent_track_ids.append(track.id)
        self.recent_track_ids = self.recent_track_ids[-12:]

    def _current_track(self) -> Track | None:
        if self.current_track_id is None:
            return None
        return self.tracks.get(self.current_track_id)

    def _track_from_spotify_state(self, state: SpotifyPlaybackState | None) -> Track | None:
        if state is None:
            return None
        if state.track_id and state.track_id in self.tracks:
            return self.tracks[state.track_id]
        if state.spotify_uri:
            for track in self.tracks.values():
                if track.spotify_uri == state.spotify_uri:
                    return track
        return None

    def _get_track(self, track_id: str) -> Track:
        try:
            return self.tracks[track_id]
        except KeyError as exc:
            raise ValueError(f"unknown track id: {track_id}") from exc

    def _validate_track_ids(self, track_ids: list[str]) -> None:
        for track_id in track_ids:
            self._get_track(track_id)

    def _queue_cap(self, track_ids: list[str]) -> list[str]:
        if self.queue_max_tracks is None:
            return list(track_ids)
        return list(track_ids[: max(1, self.queue_max_tracks)])

    @staticmethod
    def _volume_percent(value: int) -> int:
        return max(0, min(100, int(value)))

    async def _ensure_spotify_device(self) -> None:
        state = await self.spotify.get_current_playback()
        if state and state.device and state.device.id and not state.device.is_restricted:
            self.preferred_device_id = state.device.id
            return

        devices = await self.spotify.list_devices()
        candidates = [device for device in devices if device.id and not device.is_restricted]
        if not candidates:
            return

        preferred = next(
            (device for device in candidates if device.id == self.preferred_device_id),
            None,
        )
        active = next((device for device in candidates if device.is_active), None)
        selected = preferred or active or candidates[0]
        if selected.id:
            await self.spotify.transfer_playback(selected.id, play=False)
            self.preferred_device_id = selected.id

    async def _spotify_candidates(
        self,
        *,
        query: str,
        mode: str,
        avoid_clusters: set[str],
        limit: int,
    ) -> list[Track]:
        load_catalog_task = asyncio.create_task(self._load_spotify_playlist_catalog())
        search_task = asyncio.create_task(self.spotify.search_tracks(query, limit=max(1, min(limit, 12))))
        await load_catalog_task

        playlist_tracks = [
            track
            for track in self.tracks.values()
            if track.cluster.startswith("playlist:")
            and track.cluster not in avoid_clusters
            and track.id not in self.recent_track_ids
        ]
        if mode in {"shift", "adjacent_shift", "slight_shift"} and self.current_cluster:
            shifted = [track for track in playlist_tracks if track.cluster != self.current_cluster]
            playlist_tracks = shifted or playlist_tracks

        query_words = [word for word in query.lower().split() if word]
        matched_playlist_tracks = [
            track for track in playlist_tracks if self._query_match_score(track, query_words) > 0
        ]
        matched_playlist_tracks.sort(key=lambda track: self._query_match_score(track, query_words), reverse=True)

        search_tracks = await search_task
        self._register_tracks(search_tracks)

        candidates: list[Track] = []
        seen: set[str] = set()
        for track in [*matched_playlist_tracks, *search_tracks, *playlist_tracks]:
            if track.id in seen or track.cluster in avoid_clusters or track.id in self.recent_track_ids:
                continue
            candidates.append(track)
            seen.add(track.id)
            if len(candidates) >= max(1, min(limit, 12)):
                break
        return candidates

    async def _load_spotify_playlist_catalog(self) -> None:
        if self._spotify_playlist_catalog_loaded:
            return
        playlists = await self.spotify.list_user_playlists(limit=self.playlist_limit)
        playlist_track_sets = await asyncio.gather(
            *[
                self.spotify.list_playlist_tracks(
                    playlist.id,
                    playlist.name,
                    limit=self.playlist_track_limit,
                )
                for playlist in playlists
            ]
        )
        for tracks in playlist_track_sets:
            self._register_tracks(tracks)
        self._spotify_playlist_catalog_loaded = True

    def _register_tracks(self, tracks: list[Track]) -> None:
        for track in tracks:
            self.tracks[track.id] = track

    def _query_match_score(self, track: Track, query_words: list[str]) -> int:
        haystack = f"{track.title} {track.artist} {track.cluster}".lower()
        return sum(1 for word in query_words if word in haystack)


def default_demo_tracks() -> list[Track]:
    env_uris = os.environ.get("CLAUDE_DJ_DEMO_TRACK_URIS")
    if env_uris:
        return [
            Track(
                id=f"demo-track-{index}",
                title=f"Demo Track {index}",
                artist="Spotify Demo Catalog",
                spotify_uri=uri.strip(),
                cluster="demo",
            )
            for index, uri in enumerate(env_uris.split(","), start=1)
            if uri.strip()
        ]

    return [
        Track(
            id="demo-track-1",
            title="Spotify Demo Track 1",
            artist="Spotify Demo Catalog",
            spotify_uri="spotify:track:4iV5W9uYEdYUVa79Axb7Rh",
            cluster="demo",
        ),
        Track(
            id="demo-track-2",
            title="Spotify Demo Track 2",
            artist="Spotify Demo Catalog",
            spotify_uri="spotify:track:1301WleyT98MSxVHPZCA6M",
            cluster="demo",
        ),
    ]
