"""Phase 3 tests — CLAP embeddings (`embed_clap.py`).

Two layers:

1. **Always-on** tests of the pure-Python helpers (L2 normalization + JSONL
   writing + record building). These use the committed fixture embeddings and
   synthetic vectors, and never import torch/librosa/laion_clap, so they pass on
   any interpreter (including a Python build with no torch wheels).

2. **Model-gated** tests (``pytest.importorskip("laion_clap")``) that actually
   run the model on the committed fixture audio. These skip cleanly when
   laion-clap (or its torch stack) cannot be imported.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from recommendation_engine import config
from recommendation_engine.contracts import (
    EMBED_DIM,
    load_embeddings,
    load_enriched_tracks,
    validate_embedding,
)
from recommendation_engine.embed_clap import (
    embed_tracks,
    embedding_record,
    l2_normalize,
    write_embeddings_jsonl,
)


# --- Always-on: pure helpers -------------------------------------------------
def test_l2_normalize_unit_norm():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(EMBED_DIM).astype(np.float32) * 7.0
    unit = l2_normalize(raw)
    assert unit.shape == (EMBED_DIM,)
    assert unit.dtype == np.float32
    assert math.isclose(float(np.linalg.norm(unit)), 1.0, abs_tol=1e-5)


def test_l2_normalize_rejects_wrong_dim():
    with pytest.raises(ValueError):
        l2_normalize([1.0, 2.0, 3.0])


def test_l2_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        l2_normalize(np.zeros(EMBED_DIM, dtype=np.float32))


def test_l2_normalize_rejects_non_finite():
    bad = np.ones(EMBED_DIM, dtype=np.float32)
    bad[0] = np.nan
    with pytest.raises(ValueError):
        l2_normalize(bad)


def test_embedding_record_is_contract_valid():
    rng = np.random.default_rng(1)
    raw = rng.standard_normal(EMBED_DIM).astype(np.float32) * 3.0
    record = embedding_record("deezer:3135556", raw)
    # Must validate against the shared contract.
    validate_embedding(record)
    assert record["id"] == "deezer:3135556"
    assert len(record["vector"]) == EMBED_DIM
    assert math.isclose(
        math.sqrt(sum(x * x for x in record["vector"])), 1.0, abs_tol=1e-3
    )


def test_write_embeddings_jsonl_roundtrips(tmp_path):
    rng = np.random.default_rng(2)
    records = [
        embedding_record(f"deezer:{i}", rng.standard_normal(EMBED_DIM))
        for i in range(3)
    ]
    out = tmp_path / "embeddings.jsonl"
    n = write_embeddings_jsonl(records, out)
    assert n == 3
    # One JSON object per line, each contract-valid (load_embeddings validates).
    loaded = load_embeddings(out)
    assert [r["id"] for r in loaded] == [r["id"] for r in records]
    assert out.read_text().count("\n") == 3


def test_write_embeddings_jsonl_rejects_invalid(tmp_path):
    out = tmp_path / "bad.jsonl"
    with pytest.raises(ValueError):
        write_embeddings_jsonl([{"id": "deezer:1", "vector": [0.0, 1.0]}], out)


def test_committed_fixture_embeddings_validate():
    """The committed Artifact C fixture must satisfy the contract: 512-d,
    finite, L2-normalized, one record per fixture track."""
    records = load_embeddings(config.FIXTURE_EMBEDDINGS_PATH)
    enriched = load_enriched_tracks(config.FIXTURE_TRACKS_ENRICHED_PATH)
    fixture_ids = {t.id for t in enriched}
    embed_ids = {r["id"] for r in records}
    assert embed_ids == fixture_ids
    for record in records:
        validate_embedding(record)
        norm = math.sqrt(sum(x * x for x in record["vector"]))
        assert abs(norm - 1.0) <= 1e-3
        assert len(record["vector"]) == EMBED_DIM


def test_fixture_embeddings_are_normalized_helper_path():
    """Re-running our helper on the fixture vectors keeps them unit-norm and
    contract-valid (exercises l2_normalize + embedding_record on real data)."""
    for record in load_embeddings(config.FIXTURE_EMBEDDINGS_PATH):
        rebuilt = embedding_record(record["id"], record["vector"])
        validate_embedding(rebuilt)
        assert math.isclose(
            float(np.linalg.norm(rebuilt["vector"])), 1.0, abs_tol=1e-5
        )


# --- Model-gated: requires laion-clap + torch stack --------------------------
def test_embed_tracks_on_fixture_audio(tmp_path):
    """Run the real model on the committed fixture audio and assert the output
    is contract-valid: 3 lines, each a finite 512-vector with norm ~= 1.0, and
    ids covering the fixture tracks.

    Skips if laion-clap / its torch stack can't be imported, OR if the CLAP
    checkpoint can't be obtained in this environment (no local checkpoint +
    no network to download the default). The checkpoint download is a runtime
    asset, not part of this phase's code under test."""
    pytest.importorskip("laion_clap")
    pytest.importorskip("librosa")

    out = tmp_path / "embeddings.jsonl"
    try:
        written = embed_tracks(
            enriched_path=config.FIXTURE_TRACKS_ENRICHED_PATH,
            output_path=out,
            batch_size=2,
        )
    except Exception as exc:  # noqa: BLE001
        # No checkpoint on disk + no network -> can't load weights. That's an
        # environment limitation, not a code failure: treat like model-absent.
        import urllib.error

        msg = str(exc).lower()
        network_like = isinstance(
            exc, (urllib.error.URLError, OSError, ConnectionError)
        ) or any(
            tok in msg
            for tok in ("download", "certificate", "ssl", "connection", "checkpoint", "url")
        )
        if network_like:
            pytest.skip(f"CLAP checkpoint unavailable in this environment: {exc!r}")
        raise
    enriched = load_enriched_tracks(config.FIXTURE_TRACKS_ENRICHED_PATH)
    assert written == len(enriched)

    records = load_embeddings(out)  # validates every record
    assert {r["id"] for r in records} == {t.id for t in enriched}
    for record in records:
        assert len(record["vector"]) == EMBED_DIM
        assert all(math.isfinite(x) for x in record["vector"])
        norm = math.sqrt(sum(x * x for x in record["vector"]))
        assert abs(norm - 1.0) <= 1e-3
