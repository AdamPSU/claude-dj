# recommendation_engine

ClaudeDJ's emotion-aware recommendation engine. Pipeline:

```
Spotify playlist  ->  Deezer enrichment  ->  CLAP embeddings  ->  Redis vector search  ->  next_five()
   (track list)        (+ album genre,        (512-d, L2-norm)      (idx:tracks +           (same-genre /
                         30s preview mp3)                            centroids)              most-distant switch)
```

The full phase-by-phase spec is in `../IMPLEMENTATION_PLAN.md`. Data shapes are in
`contracts/engine_contracts.md` (executable form: `src/recommendation_engine/contracts.py`).

## Setup
```bash
uv sync                 # create venv + install (dev group included)
cp .env.example .env    # fill in Spotify creds, playlist id, Redis password
```

## Phase 0 (done)
```bash
uv run python scripts/make_fixtures.py   # regenerate committed fixtures
uv run pytest                            # contracts + fixtures green
```

## Full run (after later phases land)
```bash
uv run python -m recommendation_engine.scrape_spotify   # -> data/tracks_raw.json
uv run python -m recommendation_engine.enrich_deezer    # -> data/tracks_enriched.json + audio/
uv run python -m recommendation_engine.embed_clap       # -> data/embeddings.jsonl
uv run python -m recommendation_engine.store_redis      # -> Redis
uv run pytest
```

## Conventions
- `uv` only. Conventional Commits. Don't commit `.env`. Fixtures under `data/fixtures/` are committed;
  other `data/` artifacts are generated and git-ignored.

> Note: Python is pinned to 3.14. `torch`/`laion-clap` (Phase 3 only) may lack 3.14 wheels — if so,
> run the embed phase in a separate 3.11 venv inside this directory; all other phases are 3.14-clean.
