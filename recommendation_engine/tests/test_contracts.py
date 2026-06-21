"""Phase 0 acceptance: schemas, slugify, and committed-fixture validation."""

from __future__ import annotations

import hashlib

import pytest

from recommendation_engine import config
from recommendation_engine.contracts import (
    EMBED_DIM,
    load_embeddings,
    load_enriched_tracks,
    load_raw_tracks,
    slugify_genre,
    validate_enriched_track,
)


def test_slugify_genre():
    assert slugify_genre("Rap/Hip Hop") == "rap_hip_hop"
    assert slugify_genre("Dance") == "dance"
    assert slugify_genre("  R&B / Soul  ") == "r_b_soul"
    assert slugify_genre("Pop") == "pop"


def test_fixtures_exist():
    for path in (
        config.FIXTURE_TRACKS_RAW_PATH,
        config.FIXTURE_TRACKS_ENRICHED_PATH,
        config.FIXTURE_EMBEDDINGS_PATH,
    ):
        assert path.exists(), f"missing fixture {path} — run scripts/make_fixtures.py"


def test_raw_fixture_validates():
    rows = load_raw_tracks(config.FIXTURE_TRACKS_RAW_PATH)
    assert len(rows) == 3


def test_enriched_fixture_validates_and_spans_two_genres():
    rows = load_enriched_tracks(config.FIXTURE_TRACKS_ENRICHED_PATH)
    assert len(rows) == 3
    genres = {r.genre_tag for r in rows}
    assert genres == {"dance", "pop"}, "fixtures must span >=2 genres for switch tests"


def test_enriched_fixture_audio_and_hash_match():
    for r in load_enriched_tracks(config.FIXTURE_TRACKS_ENRICHED_PATH):
        mp3 = config.PACKAGE_ROOT / r.mp3_path
        assert mp3.exists(), f"missing fixture audio {mp3}"
        actual = hashlib.sha256(mp3.read_bytes()).hexdigest()
        assert actual == r.clip_hash, f"clip_hash mismatch for {r.id}"


def test_embeddings_fixture_validates():
    embs = load_embeddings(config.FIXTURE_EMBEDDINGS_PATH)
    assert len(embs) == 3
    assert all(len(e["vector"]) == EMBED_DIM for e in embs)


def test_embeddings_cover_enriched_ids():
    ids_enriched = {r.id for r in load_enriched_tracks(config.FIXTURE_TRACKS_ENRICHED_PATH)}
    ids_emb = {e["id"] for e in load_embeddings(config.FIXTURE_EMBEDDINGS_PATH)}
    assert ids_enriched == ids_emb


def test_validator_rejects_bad_genre_tag():
    rows = load_enriched_tracks(config.FIXTURE_TRACKS_ENRICHED_PATH)
    bad = rows[0].to_dict()
    bad["genre_tag"] = "WRONG"
    with pytest.raises(ValueError):
        validate_enriched_track(bad)
