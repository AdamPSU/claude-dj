"""CLI: import the last-played track and publish a seed for the harness.

    uv run python -m recommendation_engine.import_history

Reads the user's most recent track via the Spotify provider, runs the import
(Deezer -> CLAP -> Redis), and writes the resolved seed id to
``config.INITIAL_SEED_REDIS_KEY``. Requires the ``user-read-recently-played``
scope on ``SPOTIFY_REFRESH_TOKEN`` (re-authorize if it was minted earlier):

    uv run python -m recommendation_engine.scrape_spotify --authorize
"""

from __future__ import annotations

import logging
import sys

from .. import config
from .importer import import_last_played
from .spotify_provider import SpotifyHistoryProvider


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config.load_dotenv()

    provider = SpotifyHistoryProvider.from_env()
    result = import_last_played(provider)

    print(
        f"[import_history] seed={result.seed_track_id} "
        f"imported={result.imported} fell_back={result.fell_back} "
        f"reason={result.reason} "
        f"(published to {config.INITIAL_SEED_REDIS_KEY!r})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
