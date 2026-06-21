"""Import the user's last-played track and seed recommendations from it.

This is a single-track, on-demand version of the offline ingestion pipeline,
composed entirely from existing phase functions so nothing is reimplemented:

    provider.last_played()         -> ExternalTrack          (platform-agnostic)
    -> RawTrack                    (Artifact A shape)
    -> enrich_deezer.enrich_track  (Deezer match, 30s preview, album genre)
    -> embed_clap (CLAP 512-d)     (l2-normalized embedding)
    -> store_redis.store_track     (track:{deezer_id} in idx:tracks)
    -> recommend.next_five         (verify the genre has candidates)

Any failure to produce a usable seed — empty history, no Deezer match, no
preview, no album genre, embedding failure, or a genre too sparse to recommend
from — falls back to the default starting track ("Don't" by Bryson Tiller,
:data:`config.DEFAULT_SEED_TRACK_ID`).

On success the resolved seed id is published to
:data:`config.INITIAL_SEED_REDIS_KEY` so the autonomous harness can adopt it as
its initial seed at startup and serve recommendations exactly as it normally
would.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .. import config
from ..contracts import RawTrack, validate_raw_track
from ..enrich_deezer import DeezerClient, EnrichStats, enrich_track
from ..redis_raw import get_raw_redis_client
from ..recommend import next_five
from ..store_redis import store_track
from .provider import ExternalTrack, HistoryProvider

logger = logging.getLogger(__name__)

# Signal used for the freshly imported seed: we want tracks *similar* to what the
# user just listened to, which is the same-genre KNN path (positive/neutral).
IMPORT_SIGNAL = "neutral"

# The imported track is written under the indexed ``track:`` prefix so it works
# as a seed, but it should not pollute the shared catalog permanently. A TTL
# makes it ephemeral: RediSearch drops expired keys from ``idx:tracks``
# automatically, so the import self-cleans. The published seed pointer expires
# on the same clock so the harness never points at an expired track.
IMPORT_TRACK_TTL_SECONDS = 3600  # 1 hour

EmbedFn = Callable[[Path], Sequence[float]]
RecommendFn = Callable[[str, str], list[str]]


@dataclass(frozen=True)
class ImportResult:
    """Outcome of an import attempt."""

    seed_track_id: str  # "deezer:{id}" — the imported track or the default seed
    imported: bool  # True when seeding from the freshly imported track
    fell_back: bool  # True when seeding from the default track
    reason: str | None  # why we fell back (None on success)
    source: str | None  # provider name, when an ExternalTrack was obtained
    external_track: ExternalTrack | None


def default_embed_fn(checkpoint: str | None = None) -> EmbedFn:
    """Return an embed function that lazily loads CLAP once, then embeds files.

    Kept out of :func:`import_last_played`'s import-time path so tests can inject
    a lightweight stub and CLAP/torch are only loaded when actually embedding.
    Checkpoint resolution lives in :func:`embed_clap.load_model` /
    :func:`config.clap_checkpoint`.
    """
    model: Any = None

    def _embed(mp3_path: Path) -> Sequence[float]:
        nonlocal model
        from ..embed_clap import embed_files, l2_normalize, load_model

        if model is None:
            model = load_model(checkpoint)
        raw = embed_files(model, [mp3_path])
        return l2_normalize(raw[0]).tolist()

    return _embed


def _external_to_raw(ext: ExternalTrack) -> RawTrack:
    """Map a provider-neutral track onto the Artifact A shape.

    The ``spotify_id`` slot carries the provider's native id (the Spotify track
    id today). It flows through to the stored hash, where playback builds the
    Spotify URI from it — so this currently assumes the id is Spotify-resolvable,
    which holds for the Spotify provider.
    """
    return RawTrack(
        spotify_id=ext.source_id,
        title=ext.title,
        artist=ext.artist,
        isrc=ext.isrc,
        album_name=ext.album_name,
    )


def publish_initial_seed(client: Any, seed_track_id: str) -> None:
    """Publish the resolved seed id for the harness to read at startup.

    Expires on the same clock as the imported track so the pointer never
    outlives the track it names (the harness then cleanly falls back to its
    default seed).
    """
    client.set(config.INITIAL_SEED_REDIS_KEY, seed_track_id, ex=IMPORT_TRACK_TTL_SECONDS)


def import_last_played(
    provider: HistoryProvider,
    *,
    deezer_client: DeezerClient | None = None,
    embed_fn: EmbedFn | None = None,
    recommend_fn: RecommendFn | None = None,
    redis_client: Any | None = None,
    audio_dir: Path | None = None,
    publish_seed: bool = True,
    log: Callable[[str], None] = print,
) -> ImportResult:
    """Import ``provider``'s last-played track and resolve a seed track id.

    Returns an :class:`ImportResult`; never raises for the expected failure
    modes (they fall back to the default seed). Dependencies are injectable so
    this is unit-testable without network, CLAP, or a live Redis.
    """
    deezer_client = deezer_client or DeezerClient()
    embed_fn = embed_fn or default_embed_fn()
    redis_client = redis_client or get_raw_redis_client()
    if recommend_fn is None:
        recommend_fn = lambda track_id, signal: next_five(track_id, signal, client=redis_client)
    audio_dir = audio_dir or config.AUDIO_DIR

    def fall_back(reason: str, ext: ExternalTrack | None) -> ImportResult:
        log(f"[import_history] falling back to default seed ({reason})")
        if publish_seed:
            publish_initial_seed(redis_client, config.DEFAULT_SEED_TRACK_ID)
        return ImportResult(
            seed_track_id=config.DEFAULT_SEED_TRACK_ID,
            imported=False,
            fell_back=True,
            reason=reason,
            source=ext.source if ext else None,
            external_track=ext,
        )

    # 1. Last-played track from the provider.
    ext = provider.last_played()
    if ext is None:
        return fall_back("no_history", None)
    log(f"[import_history] {provider.name} last played: {ext.artist} - {ext.title}")

    # 2. Provider-neutral -> Artifact A.
    try:
        raw = validate_raw_track(_external_to_raw(ext).to_dict())
    except ValueError as exc:
        log(f"[import_history] unusable track from {provider.name}: {exc}")
        return fall_back("invalid_track", ext)

    # 3. Deezer cross-reference + 30s preview + album genre (reused as-is).
    #    enrich_track returns None for no_match / no_preview / no_genre.
    stats = EnrichStats()
    try:
        enriched = enrich_track(raw, deezer_client, audio_dir, stats, log=log)
    except Exception as exc:  # noqa: BLE001 - network/Deezer hiccup -> fall back
        log(f"[import_history] enrichment error: {exc}")
        return fall_back("enrichment_error", ext)
    if enriched is None:
        reason = (
            "no_genre" if stats.dropped_no_genre
            else "no_preview" if stats.dropped_no_preview
            else "no_match"
        )
        return fall_back(reason, ext)

    # 4. CLAP embedding of the downloaded preview (reused as-is).
    mp3_path = audio_dir / f"{enriched.deezer_id}.mp3"
    try:
        vector = list(embed_fn(mp3_path))
    except Exception as exc:  # noqa: BLE001 - embedding failure -> fall back
        log(f"[import_history] embedding error: {exc}")
        return fall_back("embedding_failed", ext)

    # 5. Store as an indexed track (reused as-is), but make it ephemeral via a
    #    TTL so it doesn't permanently pollute the shared catalog.
    store_track(redis_client, enriched, vector)
    track_key = f"{config.TRACK_KEY_PREFIX}{enriched.deezer_id}"
    redis_client.expire(track_key, IMPORT_TRACK_TTL_SECONDS)
    log(f"[import_history] stored {enriched.id} ({enriched.genre}); ttl={IMPORT_TRACK_TTL_SECONDS}s")

    # 6. Verify the genre actually has candidates to recommend from.
    try:
        candidates = recommend_fn(enriched.id, IMPORT_SIGNAL)
    except Exception as exc:  # noqa: BLE001 - retrieval issue -> fall back
        log(f"[import_history] recommendation check failed: {exc}")
        return fall_back("recommendation_check_failed", ext)
    if not candidates:
        # Track is stored, but its genre is too sparse to seed productively.
        return fall_back("no_candidates_in_genre", ext)

    # 7. Publish the imported seed for the harness.
    if publish_seed:
        publish_initial_seed(redis_client, enriched.id)
    return ImportResult(
        seed_track_id=enriched.id,
        imported=True,
        fell_back=False,
        reason=None,
        source=ext.source,
        external_track=ext,
    )
