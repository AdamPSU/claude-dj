"""Artifact contracts shared across all phases.

This module is the single source of truth for the shape of the data that flows
between phases. Each phase must validate its output here before exiting.

Artifacts:
  A  RawTrack          -> data/tracks_raw.json        (Phase 1 -> Phase 2)
  B  EnrichedTrack     -> data/tracks_enriched.json   (Phase 2 -> Phase 3, 4)
  C  Embedding (dict)  -> data/embeddings.jsonl        (Phase 3 -> Phase 4)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

EMBED_DIM = 512
VALID_MATCH_METHODS = {"isrc", "search"}
UNIT_NORM_TOLERANCE = 1e-3


# --- Genre slug --------------------------------------------------------------
def slugify_genre(name: str) -> str:
    """Deezer genres contain '/' and spaces (e.g. 'Rap/Hip Hop') which break
    RediSearch TAG queries. Slugify to a safe, stable tag: 'rap_hip_hop'."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# --- Artifact A: raw track from Spotify --------------------------------------
@dataclass
class RawTrack:
    spotify_id: str
    title: str
    artist: str
    isrc: str  # uppercase; may be empty if Spotify had none
    album_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Artifact B: enriched, genre-bearing track from Deezer -------------------
@dataclass
class EnrichedTrack:
    id: str  # "deezer:{deezer_id}"
    deezer_id: str
    spotify_id: str
    isrc: str
    title: str
    artist: str
    album: str
    genre: str  # display, e.g. "Rap/Hip Hop"
    genre_tag: str  # slug, e.g. "rap_hip_hop" (the indexed TAG)
    artwork_url: str
    preview_source: str  # always "deezer"
    duration_seconds: int
    rank: int
    mp3_path: str  # relative to the recommendation_engine/ root
    clip_hash: str  # sha256 hex of the downloaded mp3 bytes
    match_method: str  # "isrc" | "search"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Validators (raise ValueError with a precise message) --------------------
def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _check_fields(data: dict[str, Any], dc: type) -> None:
    expected = {f.name for f in fields(dc)}
    missing = expected - data.keys()
    extra = data.keys() - expected
    _require(not missing, f"{dc.__name__} missing fields: {sorted(missing)}")
    _require(not extra, f"{dc.__name__} unexpected fields: {sorted(extra)}")


def validate_raw_track(data: dict[str, Any]) -> RawTrack:
    _check_fields(data, RawTrack)
    for key in ("spotify_id", "title", "artist", "album_name"):
        _require(isinstance(data[key], str) and data[key].strip() != "",
                 f"RawTrack.{key} must be a non-empty string")
    _require(isinstance(data["isrc"], str), "RawTrack.isrc must be a string (may be empty)")
    return RawTrack(**data)


def validate_enriched_track(data: dict[str, Any]) -> EnrichedTrack:
    _check_fields(data, EnrichedTrack)
    str_required = ("id", "deezer_id", "title", "artist", "album", "genre",
                    "genre_tag", "mp3_path", "clip_hash")
    for key in str_required:
        _require(isinstance(data[key], str) and data[key].strip() != "",
                 f"EnrichedTrack.{key} must be a non-empty string")
    _require(data["id"] == f"deezer:{data['deezer_id']}",
             f"EnrichedTrack.id must equal 'deezer:{{deezer_id}}' (got {data['id']!r})")
    _require(data["genre_tag"] == slugify_genre(data["genre"]),
             f"EnrichedTrack.genre_tag must be slugify_genre(genre); "
             f"got {data['genre_tag']!r} for genre {data['genre']!r}")
    _require(data["preview_source"] == "deezer",
             "EnrichedTrack.preview_source must be 'deezer'")
    _require(data["match_method"] in VALID_MATCH_METHODS,
             f"EnrichedTrack.match_method must be one of {VALID_MATCH_METHODS}")
    _require(isinstance(data["duration_seconds"], int) and data["duration_seconds"] > 0,
             "EnrichedTrack.duration_seconds must be a positive int")
    _require(isinstance(data["rank"], int) and data["rank"] >= 0,
             "EnrichedTrack.rank must be a non-negative int")
    return EnrichedTrack(**data)


def validate_embedding(data: dict[str, Any]) -> dict[str, Any]:
    _require(set(data.keys()) == {"id", "vector"},
             f"Embedding must have exactly keys id, vector (got {sorted(data.keys())})")
    _require(isinstance(data["id"], str) and data["id"].startswith("deezer:"),
             "Embedding.id must be a 'deezer:...' string")
    vec = data["vector"]
    _require(isinstance(vec, list) and len(vec) == EMBED_DIM,
             f"Embedding.vector must be a list of {EMBED_DIM} floats (got len {len(vec) if isinstance(vec, list) else 'n/a'})")
    _require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in vec),
             "Embedding.vector must contain only finite numbers")
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    _require(abs(norm - 1.0) <= UNIT_NORM_TOLERANCE,
             f"Embedding.vector must be L2-normalized (norm={norm:.6f}, tol={UNIT_NORM_TOLERANCE})")
    return data


# --- File loaders (validate every record) ------------------------------------
def load_raw_tracks(path: Path) -> list[RawTrack]:
    records = json.loads(Path(path).read_text())
    _require(isinstance(records, list), "tracks_raw.json must be a JSON array")
    return [validate_raw_track(r) for r in records]


def load_enriched_tracks(path: Path) -> list[EnrichedTrack]:
    records = json.loads(Path(path).read_text())
    _require(isinstance(records, list), "tracks_enriched.json must be a JSON array")
    return [validate_enriched_track(r) for r in records]


def load_embeddings(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"embeddings.jsonl line {i + 1} is not valid JSON: {exc}") from exc
        out.append(validate_embedding(record))
    return out


def dump_json(records: list[Any], path: Path) -> None:
    """Write a list of dataclasses or dicts as a JSON array."""
    payload = [r.to_dict() if hasattr(r, "to_dict") else r for r in records]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
