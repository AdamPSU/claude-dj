"""Generate the committed fixtures under data/fixtures/.

Run once (re-runnable / deterministic):
    uv run python scripts/make_fixtures.py

Produces 3 tracks across 2 genres (2x dance, 1x pop) so downstream phases can
exercise both the same-genre and switch-genre paths in isolation. The pop vector
is deliberately distant from the two dance vectors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from recommendation_engine import config
from recommendation_engine.contracts import (
    EMBED_DIM,
    EnrichedTrack,
    RawTrack,
    dump_json,
    slugify_genre,
)

SR = 44_100
DURATION_S = 1.0

# (deezer_id, spotify_id, title, artist, album, genre, isrc, rank, duration_s, tone_hz, family)
TRACKS = [
    ("3135556", "6Qyc6fS4DsZjB2mRW9DsQs", "Harder, Better, Faster, Stronger",
     "Daft Punk", "Discovery", "Dance", "GBDUW0000059", 814839, 224, 440.0, "dance"),
    ("14552159", "1pKYYY0dkg23sQQXi0Q5zN", "Night Drive",
     "Synth Collective", "Neon", "Dance", "FRUM71500001", 512345, 198, 523.25, "dance"),
    ("67238735", "0VjIjW4GlUZAMYd2vXMi3b", "Golden Hour",
     "The Lights", "Daybreak", "Pop", "USUM71700002", 623410, 211, 659.25, "pop"),
]


def write_tone(path: Path, hz: float) -> str:
    """Write a short sine-tone MP3 and return its sha256 hex."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, DURATION_S, int(SR * DURATION_S), endpoint=False)
    wave = (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    sf.write(str(path), wave, SR, format="MP3")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_vector(rng: np.random.Generator, base: np.ndarray) -> list[float]:
    v = base + 0.05 * rng.standard_normal(EMBED_DIM)
    v = v / np.linalg.norm(v)
    return [float(x) for x in v.astype(np.float32)]


def main() -> None:
    config.FIXTURE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    base = {
        "dance": rng.standard_normal(EMBED_DIM),
        "pop": rng.standard_normal(EMBED_DIM),
    }

    raw: list[RawTrack] = []
    enriched: list[EnrichedTrack] = []
    embeddings: list[dict] = []

    for (deezer_id, spotify_id, title, artist, album, genre, isrc,
         rank, duration_s, hz, family) in TRACKS:
        # mp3 path is stored relative to the package root for portability.
        rel_mp3 = Path("data/fixtures/audio") / f"{deezer_id}.mp3"
        abs_mp3 = config.PACKAGE_ROOT / rel_mp3
        clip_hash = write_tone(abs_mp3, hz)

        raw.append(RawTrack(
            spotify_id=spotify_id, title=title, artist=artist,
            isrc=isrc, album_name=album,
        ))
        enriched.append(EnrichedTrack(
            id=f"deezer:{deezer_id}", deezer_id=deezer_id, spotify_id=spotify_id,
            isrc=isrc, title=title, artist=artist, album=album,
            genre=genre, genre_tag=slugify_genre(genre),
            artwork_url=f"https://example.test/{deezer_id}.jpg",
            preview_source="deezer", duration_seconds=duration_s, rank=rank,
            mp3_path=str(rel_mp3), clip_hash=clip_hash, match_method="isrc",
        ))
        embeddings.append({
            "id": f"deezer:{deezer_id}",
            "vector": make_vector(rng, base[family]),
        })

    dump_json(raw, config.FIXTURE_TRACKS_RAW_PATH)
    dump_json(enriched, config.FIXTURE_TRACKS_ENRICHED_PATH)
    config.FIXTURE_EMBEDDINGS_PATH.write_text(
        "\n".join(json.dumps(e) for e in embeddings) + "\n"
    )

    print(f"Wrote {len(raw)} fixtures to {config.FIXTURES_DIR}")


if __name__ == "__main__":
    main()
