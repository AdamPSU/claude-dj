"""ClaudeDJ recommendation engine.

Pipeline: Spotify playlist -> Deezer enrichment -> CLAP embeddings -> Redis vector
search -> recommendations. See IMPLEMENTATION_PLAN.md at the repo root for the full
phase-by-phase contract.
"""

from recommendation_engine.contracts import (
    EMBED_DIM,
    EnrichedTrack,
    RawTrack,
    slugify_genre,
)

__all__ = ["EMBED_DIM", "EnrichedTrack", "RawTrack", "slugify_genre"]
