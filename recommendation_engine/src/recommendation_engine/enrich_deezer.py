"""Phase 2 — Deezer enrichment, preview download, and genre filter.

Reads Artifact A (`data/tracks_raw.json`, or the committed fixture) and produces
Artifact B (`data/tracks_enriched.json` + `data/audio/{deezer_id}.mp3`).

Per track:
  1. Match to a Deezer track by ISRC (`/track/isrc:{isrc}`), falling back to
     `/search`. No match -> drop.
  2. Read deezer_id, duration, rank, preview URL, artwork, album id/title, artist.
     No preview -> drop.
  3. Fetch the album genre via `/album/{id}` (`genres.data[0].name`); album
     lookups are cached by album id. No genre -> drop (no download, no embed).
  4. Download the 30s preview and record its sha256 clip hash.
  5. Emit a validated Artifact B row.

Deezer is rate-limited (~50 req / 5s) and returns errors as HTTP-200
`{"error": {...}}` payloads. Both are handled here. The HTTP layer is funnelled
through a small client so tests can mock it without real network access.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import requests

from . import config
from .contracts import (
    EnrichedTrack,
    RawTrack,
    dump_json,
    load_raw_tracks,
    slugify_genre,
    validate_enriched_track,
)

# Retry/backoff for transient Deezer failures (429 / 5xx).
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 30


class DeezerError(RuntimeError):
    """Raised when Deezer returns an error payload (HTTP-200 {"error": ...})."""


class RateLimiter:
    """Sliding-window token limiter: at most ``max_requests`` per ``window``."""

    def __init__(self, max_requests: int, window_seconds: float,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._sleep = sleep
        self._clock = clock
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = self._clock()
        # Drop timestamps that have aged out of the window.
        while self._calls and now - self._calls[0] >= self.window_seconds:
            self._calls.popleft()
        if len(self._calls) >= self.max_requests:
            wait = self.window_seconds - (now - self._calls[0])
            if wait > 0:
                self._sleep(wait)
            now = self._clock()
            while self._calls and now - self._calls[0] >= self.window_seconds:
                self._calls.popleft()
        self._calls.append(self._clock())


class DeezerClient:
    """Thin Deezer API wrapper. The two network seams (`get_json`, `download`)
    are the only things tests need to mock."""

    def __init__(self, base_url: str = config.DEEZER_BASE_URL,
                 session: requests.Session | None = None,
                 limiter: RateLimiter | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.limiter = limiter or RateLimiter(
            config.DEEZER_MAX_REQUESTS, config.DEEZER_WINDOW_SECONDS, sleep=sleep,
        )
        self._sleep = sleep
        # Album genre cache, keyed by album id. Value is the resolved
        # (genre, genre_tag) tuple or None when the album has no genre.
        self._album_cache: dict[str, tuple[str, str] | None] = {}

    # -- low-level network seams ------------------------------------------
    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET {base}/{path} and return parsed JSON.

        Treats HTTP-200 ``{"error": ...}`` payloads as errors and retries
        transient HTTP failures (429 / 5xx) with backoff.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self.limiter.acquire()
            resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            status = resp.status_code
            if status == 429 or status >= 500:
                last_exc = DeezerError(f"transient HTTP {status} for {url}")
                self._sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                err = data["error"]
                # Deezer quota errors arrive as 200 {"error": {"code": 4, ...}}.
                code = err.get("code") if isinstance(err, dict) else None
                if code == 4:  # quota exceeded -> back off and retry
                    last_exc = DeezerError(f"quota error for {url}: {err}")
                    self._sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
                    continue
                raise DeezerError(f"Deezer error for {url}: {err}")
            return data
        raise last_exc or DeezerError(f"exhausted retries for {url}")

    def download(self, url: str) -> bytes:
        """Download raw bytes (used for the 30s preview MP3)."""
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self.limiter.acquire()
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = DeezerError(f"transient HTTP {resp.status_code} for {url}")
                self._sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp.content
        raise last_exc or DeezerError(f"exhausted retries downloading {url}")

    # -- domain operations ------------------------------------------------
    def match_track(self, track: RawTrack) -> tuple[dict[str, Any] | None, str]:
        """Return (deezer_track_object, match_method).

        Tries ISRC first (when present), then a search fallback. Returns
        (None, "") when nothing matches.
        """
        if track.isrc:
            try:
                obj = self.get_json(f"/track/isrc:{track.isrc}")
                if obj and obj.get("id"):
                    return obj, "isrc"
            except DeezerError:
                pass  # fall through to search
        query = f'artist:"{track.artist}" track:"{track.title}"'
        try:
            results = self.get_json("/search", params={"q": query})
        except DeezerError:
            return None, ""
        data = results.get("data") if isinstance(results, dict) else None
        if data:
            return data[0], "search"
        return None, ""

    def album_genre(self, album_id: str) -> tuple[str, str] | None:
        """Return (genre, genre_tag) for an album, or None when it has no genre.

        Results are cached by album id so albums shared across tracks are
        fetched at most once.
        """
        key = str(album_id)
        if key in self._album_cache:
            return self._album_cache[key]
        try:
            album = self.get_json(f"/album/{key}")
        except DeezerError:
            self._album_cache[key] = None
            return None
        genres = (album.get("genres") or {}).get("data") or []
        result: tuple[str, str] | None = None
        if genres:
            name = genres[0].get("name", "").strip()
            if name:
                result = (name, slugify_genre(name))
        self._album_cache[key] = result
        return result


@dataclass
class EnrichStats:
    total: int = 0
    enriched: int = 0
    dropped_no_match: int = 0
    dropped_no_preview: int = 0
    dropped_no_genre: int = 0

    def summary(self) -> str:
        return (
            f"enriched={self.enriched} of {self.total} | "
            f"dropped: no_match={self.dropped_no_match}, "
            f"no_preview={self.dropped_no_preview}, "
            f"no_genre={self.dropped_no_genre}"
        )


def _relative_mp3_path(mp3_file) -> str:
    """Return the mp3 path relative to the package root when possible.

    Real runs write under ``config.AUDIO_DIR`` (inside the package), giving the
    contract-specified ``data/audio/{id}.mp3``. Tests using an out-of-tree
    ``tmp_path`` fall back to the absolute posix path (still a valid non-empty
    string for the contract)."""
    try:
        return mp3_file.relative_to(config.PACKAGE_ROOT).as_posix()
    except ValueError:
        return mp3_file.as_posix()


def _artwork_url(deezer_track: dict[str, Any]) -> str:
    album = deezer_track.get("album") or {}
    for key in ("cover_xl", "cover_big", "cover_medium", "cover"):
        if album.get(key):
            return album[key]
    return deezer_track.get("md5_image", "") or ""


def enrich_track(track: RawTrack, client: DeezerClient,
                 audio_dir, stats: EnrichStats,
                 log: Callable[[str], None] = print) -> EnrichedTrack | None:
    """Enrich a single raw track, downloading its preview. Returns the
    Artifact B row, or None if the track is dropped."""
    deezer_track, match_method = client.match_track(track)
    if not deezer_track:
        stats.dropped_no_match += 1
        log(f"DROP no_match: {track.artist} - {track.title}")
        return None

    deezer_id = str(deezer_track.get("id"))
    preview = deezer_track.get("preview") or ""
    if not preview:
        stats.dropped_no_preview += 1
        log(f"DROP no_preview: {track.artist} - {track.title} (deezer:{deezer_id})")
        return None

    album = deezer_track.get("album") or {}
    album_id = album.get("id")
    genre_pair = client.album_genre(album_id) if album_id is not None else None
    if not genre_pair:
        stats.dropped_no_genre += 1
        log(f"DROP no_genre: {track.artist} - {track.title} (deezer:{deezer_id})")
        return None
    genre, genre_tag = genre_pair

    # Download the preview and hash its bytes.
    audio_bytes = client.download(preview)
    clip_hash = hashlib.sha256(audio_bytes).hexdigest()
    audio_dir.mkdir(parents=True, exist_ok=True)
    mp3_file = audio_dir / f"{deezer_id}.mp3"
    mp3_file.write_bytes(audio_bytes)
    mp3_rel = _relative_mp3_path(mp3_file)

    artist = (deezer_track.get("artist") or {}).get("name") or track.artist

    row = EnrichedTrack(
        id=f"deezer:{deezer_id}",
        deezer_id=deezer_id,
        spotify_id=track.spotify_id,
        isrc=track.isrc,
        title=deezer_track.get("title") or track.title,
        artist=artist,
        album=album.get("title") or track.album_name,
        genre=genre,
        genre_tag=genre_tag,
        artwork_url=_artwork_url(deezer_track),
        preview_source="deezer",
        duration_seconds=int(deezer_track.get("duration") or 0),
        rank=int(deezer_track.get("rank") or 0),
        mp3_path=mp3_rel,
        clip_hash=clip_hash,
        match_method=match_method,
    )
    validate_enriched_track(row.to_dict())
    stats.enriched += 1
    return row


def enrich_tracks(tracks: Iterable[RawTrack], client: DeezerClient | None = None,
                  audio_dir=None,
                  log: Callable[[str], None] = print) -> tuple[list[EnrichedTrack], EnrichStats]:
    """Enrich every raw track, returning (surviving rows, stats)."""
    client = client or DeezerClient()
    audio_dir = audio_dir or config.AUDIO_DIR
    stats = EnrichStats()
    rows: list[EnrichedTrack] = []
    for track in tracks:
        stats.total += 1
        row = enrich_track(track, client, audio_dir, stats, log=log)
        if row is not None:
            rows.append(row)
    return rows, stats


def main(input_path=None, output_path=None) -> None:
    """CLI entry point: read Artifact A, write Artifact B."""
    config.load_dotenv()
    input_path = input_path or config.TRACKS_RAW_PATH
    if not input_path.exists():
        input_path = config.FIXTURE_TRACKS_RAW_PATH
    output_path = output_path or config.TRACKS_ENRICHED_PATH

    raw_tracks = load_raw_tracks(input_path)
    rows, stats = enrich_tracks(raw_tracks)
    dump_json(rows, output_path)

    # Validate the written artifact end-to-end.
    from .contracts import load_enriched_tracks
    load_enriched_tracks(output_path)

    print(f"[enrich] {stats.summary()}")
    print(f"[enrich] wrote {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
