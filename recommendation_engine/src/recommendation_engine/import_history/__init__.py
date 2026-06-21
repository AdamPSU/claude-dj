"""Platform-agnostic listening-history import.

Public surface::

    from recommendation_engine.import_history import (
        ExternalTrack, HistoryProvider, SpotifyHistoryProvider,
        ImportResult, import_last_played,
    )
"""

from __future__ import annotations

from .importer import (
    ImportResult,
    default_embed_fn,
    import_last_played,
    publish_initial_seed,
)
from .provider import ExternalTrack, HistoryProvider
from .spotify_provider import SpotifyHistoryProvider

__all__ = [
    "ExternalTrack",
    "HistoryProvider",
    "SpotifyHistoryProvider",
    "ImportResult",
    "import_last_played",
    "default_embed_fn",
    "publish_initial_seed",
]
