"""Shared configuration: paths, constants, and lightweight env loading.

No third-party dependencies — readable by every phase.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# config.py lives at: <root>/src/recommendation_engine/config.py
PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # the recommendation_engine/ dir
DATA_DIR = PACKAGE_ROOT / "data"
FIXTURES_DIR = DATA_DIR / "fixtures"
AUDIO_DIR = DATA_DIR / "audio"
MODELS_DIR = PACKAGE_ROOT / "models"

# Generated artifact paths (real runs).
TRACKS_RAW_PATH = DATA_DIR / "tracks_raw.json"
TRACKS_ENRICHED_PATH = DATA_DIR / "tracks_enriched.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.jsonl"

# Fixture artifact paths (committed; used for isolated phase development).
FIXTURE_TRACKS_RAW_PATH = FIXTURES_DIR / "tracks_raw.json"
FIXTURE_TRACKS_ENRICHED_PATH = FIXTURES_DIR / "tracks_enriched.json"
FIXTURE_EMBEDDINGS_PATH = FIXTURES_DIR / "embeddings.jsonl"
FIXTURE_AUDIO_DIR = FIXTURES_DIR / "audio"

# --- Embedding / model -------------------------------------------------------
EMBED_DIM = 512
CLAP_AMODEL = "HTSAT-tiny"
DEFAULT_CLAP_CHECKPOINT = "music_audioset_epoch_15_esc_90.14.pt"
# Legacy import-time constant. Prefer clap_checkpoint() so .env values loaded
# after import are honored.
CLAP_CHECKPOINT = os.getenv("CLAP_CHECKPOINT", DEFAULT_CLAP_CHECKPOINT)
TARGET_SAMPLE_RATE = 48_000  # CLAP expects 48 kHz mono

# --- Redis schema ------------------------------------------------------------
REDIS_INDEX = "idx:tracks"
TRACK_KEY_PREFIX = "track:"
CENTROID_KEY_PREFIX = "genre_centroid:"

# Default starting track ("Don't" by Bryson Tiller) used when no usable
# history seed can be imported. Must stay in sync with the backend harness.
DEFAULT_SEED_TRACK_ID = "deezer:100814018"
INITIAL_SEED_REDIS_KEY = "claudedj:initial_seed_track_id"

# --- Deezer ------------------------------------------------------------------
DEEZER_BASE_URL = "https://api.deezer.com"
DEEZER_MAX_REQUESTS = 50
DEEZER_WINDOW_SECONDS = 5

# --- Spotify -----------------------------------------------------------------
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no dependency). Existing env vars win."""
    env_path = path or (PACKAGE_ROOT / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def getenv(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def clap_checkpoint(explicit: str | None = None) -> str | None:
    """Resolve the CLAP checkpoint at call time, after dotenv can load."""
    load_dotenv()
    value = explicit or os.getenv("CLAP_CHECKPOINT") or DEFAULT_CLAP_CHECKPOINT
    if not value:
        return None
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(PACKAGE_ROOT / raw)
    candidates.append(MODELS_DIR / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return value
