"""Platform-agnostic listening-history provider abstraction.

The import-history feature seeds recommendations from the *last track a user
listened to* on some external service. The only platform-specific part is
*learning what that track was* — once we have a provider-neutral
:class:`ExternalTrack`, the rest of the flow (Deezer cross-reference, CLAP
embedding, Redis storage, KNN) is identical to the offline ingestion pipeline.

To add a new provider (e.g. YouTube Music), implement :class:`HistoryProvider`
returning an :class:`ExternalTrack`; nothing downstream changes. Note that
playback itself remains Spotify-based, so a non-Spotify provider's tracks still
need to resolve to a Spotify-playable track via Deezer's ISRC/name matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ExternalTrack:
    """A provider-neutral reference to one track from a user's history.

    These are exactly the fields the Deezer enrichment step needs (it matches
    on ISRC first, then artist/title), plus provenance (``source`` /
    ``source_id``) so the origin platform is never lost.
    """

    title: str
    artist: str
    isrc: str  # uppercase; "" when the provider exposes none
    album_name: str
    source: str  # provider name, e.g. "spotify", "youtube_music"
    source_id: str  # the provider's native track id (may be "")


@runtime_checkable
class HistoryProvider(Protocol):
    """A source of the user's most recently played track."""

    name: str

    def last_played(self) -> ExternalTrack | None:
        """Return the most recently played track, or None if history is empty."""
        ...
