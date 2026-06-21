"""Phase 3 — CLAP audio embeddings (`embed_clap.py`).

Reads Artifact B (`data/tracks_enriched.json` or the committed
`data/fixtures/tracks_enriched.json`), loads the LAION-CLAP *music* checkpoint
(amodel ``HTSAT-base``, 512-dim), resamples each preview mp3 to 48 kHz mono,
produces one L2-normalized 512-vector per track, and writes Artifact C
(`data/embeddings.jsonl`), validated against the contract.

Heavy imports (``laion_clap``, ``librosa``, ``torch``) are intentionally **lazy**
(performed inside the functions that need them) so this module imports cleanly in
environments where those packages are absent — e.g. a Python build without
``torch`` wheels. The pure-Python helpers below (L2 normalization, jsonl writing)
have no third-party requirements beyond ``numpy``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import config
from .contracts import (
    EMBED_DIM,
    load_enriched_tracks,
    validate_embedding,
)

logger = logging.getLogger(__name__)

# Embed in small batches to bound memory; CPU is fine for ~150 clips.
DEFAULT_BATCH_SIZE = 16


# --- Pure helpers (no torch/librosa needed) ----------------------------------
def l2_normalize(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return ``vector`` as a float32 unit vector.

    Raises ``ValueError`` if the vector has the wrong length or a (near) zero
    norm (which cannot be normalized to unit length).
    """
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vec.shape[0] != EMBED_DIM:
        raise ValueError(
            f"expected a {EMBED_DIM}-d vector, got {vec.shape[0]}"
        )
    if not np.all(np.isfinite(vec)):
        raise ValueError("vector contains non-finite values")
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"vector has zero/degenerate norm ({norm!r}); cannot normalize")
    return (vec / norm).astype(np.float32)


def embedding_record(track_id: str, vector: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Build a contract-shaped Artifact C record from an id + raw vector.

    The vector is L2-normalized and validated against ``validate_embedding``
    before being returned, so a returned record is always contract-valid.
    """
    unit = l2_normalize(vector)
    record = {"id": track_id, "vector": [float(x) for x in unit.tolist()]}
    validate_embedding(record)
    return record


def write_embeddings_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    """Write Artifact C records as JSONL. Returns the number of lines written.

    Each record is validated against the contract before being written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            validate_embedding(record)
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def _resolve_mp3_path(mp3_path: str) -> Path:
    """Resolve an Artifact B ``mp3_path`` (relative to the package root)."""
    p = Path(mp3_path)
    if not p.is_absolute():
        p = config.PACKAGE_ROOT / p
    return p


# --- Model loading + embedding (lazy heavy imports) --------------------------
def load_model(checkpoint: str | None = None, amodel: str = config.CLAP_AMODEL):
    """Load the LAION-CLAP music checkpoint once and return the model.

    Imports ``laion_clap`` lazily so this module stays importable without torch.
    """
    import laion_clap  # noqa: PLC0415 — intentional lazy import

    ckpt = checkpoint or config.CLAP_CHECKPOINT
    logger.info("loading LAION-CLAP model amodel=%s checkpoint=%s", amodel, ckpt)
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel=amodel)
    # If a checkpoint path is given and exists, load it; otherwise fall back to
    # laion-clap's default download for the given amodel.
    ckpt_path = Path(ckpt)
    if ckpt_path.exists():
        model.load_ckpt(str(ckpt_path))
    else:
        logger.warning(
            "checkpoint %r not found on disk; using laion-clap default download "
            "for amodel=%s",
            ckpt,
            amodel,
        )
        model.load_ckpt()
    return model


def embed_files(model, paths: Sequence[Path | str]) -> np.ndarray:
    """Return raw (un-normalized) embeddings for ``paths`` as an (N, 512) array.

    Resamples each clip to ``config.TARGET_SAMPLE_RATE`` (48 kHz) mono via
    librosa, then runs CLAP. Imports ``librosa`` lazily.
    """
    import librosa  # noqa: PLC0415 — intentional lazy import

    audio_batch = []
    for path in paths:
        # Resample to 48 kHz mono — CLAP's expected input rate.
        samples, _ = librosa.load(str(path), sr=config.TARGET_SAMPLE_RATE, mono=True)
        audio_batch.append(samples.astype(np.float32))

    # laion-clap accepts a list of np arrays (data) when use_tensor=False; it
    # also handles ragged-length clips by internal padding/cropping.
    embeddings = model.get_audio_embedding_from_data(x=audio_batch, use_tensor=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.shape[1] != EMBED_DIM:
        raise ValueError(
            f"model returned dim {embeddings.shape[1]}, expected {EMBED_DIM}; "
            f"check amodel/checkpoint match"
        )
    return embeddings


def _chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def embed_tracks(
    enriched_path: Path | None = None,
    output_path: Path | None = None,
    *,
    checkpoint: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Run Phase 3 end-to-end: Artifact B -> Artifact C.

    Returns the number of embedding records written. Tracks whose mp3 is missing
    or unreadable are skipped (logged), not fatal.
    """
    enriched_path = Path(enriched_path) if enriched_path else config.TRACKS_ENRICHED_PATH
    output_path = Path(output_path) if output_path else config.EMBEDDINGS_PATH

    tracks = load_enriched_tracks(enriched_path)
    logger.info("loaded %d enriched tracks from %s", len(tracks), enriched_path)

    # Drop tracks whose audio is missing up front, keep (id, path) pairs.
    usable: list[tuple[str, Path]] = []
    for track in tracks:
        mp3 = _resolve_mp3_path(track.mp3_path)
        if not mp3.exists():
            logger.warning("skipping %s: mp3 not found at %s", track.id, mp3)
            continue
        usable.append((track.id, mp3))

    if not usable:
        logger.warning("no usable audio files; writing empty %s", output_path)
        return write_embeddings_jsonl([], output_path)

    model = load_model(checkpoint=checkpoint)

    records: list[dict[str, Any]] = []
    for batch in _chunked(usable, batch_size):
        ids = [tid for tid, _ in batch]
        paths = [p for _, p in batch]
        try:
            raw = embed_files(model, paths)
        except Exception:  # noqa: BLE001 — isolate a bad batch, keep going clip-by-clip
            logger.exception("batch embed failed; retrying clip-by-clip")
            for tid, p in batch:
                try:
                    raw_one = embed_files(model, [p])
                    records.append(embedding_record(tid, raw_one[0]))
                except Exception:  # noqa: BLE001
                    logger.exception("skipping %s: failed to embed %s", tid, p)
            continue
        for tid, vec in zip(ids, raw):
            try:
                records.append(embedding_record(tid, vec))
            except ValueError:
                logger.exception("skipping %s: invalid embedding", tid)

    written = write_embeddings_jsonl(records, output_path)
    logger.info("wrote %d embeddings to %s", written, output_path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 — CLAP embeddings")
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="read the committed fixtures instead of generated artifacts",
    )
    parser.add_argument("--enriched", type=Path, default=None, help="Artifact B path")
    parser.add_argument("--output", type=Path, default=None, help="Artifact C output path")
    parser.add_argument("--checkpoint", default=None, help="CLAP checkpoint path override")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    enriched = args.enriched
    output = args.output
    if args.fixtures:
        enriched = enriched or config.FIXTURE_TRACKS_ENRICHED_PATH
        output = output or config.EMBEDDINGS_PATH

    written = embed_tracks(
        enriched_path=enriched,
        output_path=output,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
    )
    print(f"wrote {written} embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
