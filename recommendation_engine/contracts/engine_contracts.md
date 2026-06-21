# Engine contracts (Phase 0 — source of truth)

All phases read/write through these shapes. The authoritative, executable version is
`src/recommendation_engine/contracts.py`; this file is the human summary. If they ever
disagree, `contracts.py` wins — fix this doc.

## Identifiers
- `deezer_id`: Deezer's numeric track id, as a string.
- `id`: `"deezer:" + deezer_id`.
- Redis track key: `track:{deezer_id}`.
- Redis centroid key: `genre_centroid:{genre_tag}`.

## Genre: display vs tag
Deezer genres contain `/` and spaces (e.g. `"Rap/Hip Hop"`), which break RediSearch TAG
queries. Every track carries both:
- `genre` — display string, e.g. `"Rap/Hip Hop"`.
- `genre_tag` — `slugify_genre(genre)`, e.g. `"rap_hip_hop"`. **This is the indexed TAG and
  the centroid key suffix.** Rule: `[^a-z0-9]+` → `_`, lowercased, trimmed of leading/trailing `_`.

## Artifact A — `data/tracks_raw.json` (Phase 1 → Phase 2)
JSON array of:
| field | type | notes |
| --- | --- | --- |
| spotify_id | str | non-empty |
| title | str | non-empty |
| artist | str | first artist |
| isrc | str | uppercase; may be `""` |
| album_name | str | non-empty |

## Artifact B — `data/tracks_enriched.json` + `data/audio/{deezer_id}.mp3` (Phase 2 → 3, 4)
One row **only** for surviving tracks (genre present + preview downloaded):
| field | type | notes |
| --- | --- | --- |
| id | str | `deezer:{deezer_id}` |
| deezer_id | str | non-empty |
| spotify_id | str | |
| isrc | str | |
| title | str | non-empty |
| artist | str | non-empty |
| album | str | non-empty |
| genre | str | non-empty display genre |
| genre_tag | str | must equal `slugify_genre(genre)` |
| artwork_url | str | |
| preview_source | str | always `"deezer"` |
| duration_seconds | int | > 0 |
| rank | int | ≥ 0 |
| mp3_path | str | relative to `recommendation_engine/` root |
| clip_hash | str | sha256 hex of the mp3 bytes |
| match_method | str | `"isrc"` or `"search"` |

## Artifact C — `data/embeddings.jsonl` (Phase 3 → Phase 4)
One JSON object per line:
```json
{"id": "deezer:3135556", "vector": [/* 512 finite floats, L2-normalized (norm≈1) */]}
```

## Redis schema (Phase 4 → Phase 5)
Hash `track:{deezer_id}` fields: `title, artist, album, genre, genre_tag, isrc, deezer_id,
spotify_id, artwork_url, duration_seconds, rank, embedding` where `embedding` is
`numpy.float32` bytes of length `512*4 = 2048`.

```
FT.CREATE idx:tracks ON HASH PREFIX 1 track: SCHEMA
  genre_tag TAG  artist TAG  title TEXT
  embedding VECTOR FLAT 6 TYPE FLOAT32 DIM 512 DISTANCE_METRIC COSINE
```

Centroid `genre_centroid:{genre_tag}` fields: `genre_tag`, `count`, `embedding`
(float32 bytes, L2-normalized mean of that genre's vectors).

## Recommendation API (Phase 5)
```python
next_five(current_track_id: str,
          signal: str,                       # "positive" | "neutral" | "negative"
          recently_played: list[str] = None  # ids to exclude
          ) -> list[str]                       # up to 5 ids, best (most similar) first
```
- positive/neutral → KNN within the current `genre_tag`.
- negative → KNN within the genre whose centroid is most distant (cosine) from the current vector.
- Always exclude `current_track_id` and `recently_played`; ranking = cosine similarity.

## Validation helpers (use these; do not re-implement)
`contracts.py` exposes: `slugify_genre`, `validate_raw_track`, `validate_enriched_track`,
`validate_embedding`, `load_raw_tracks`, `load_enriched_tracks`, `load_embeddings`, `dump_json`.

## Fixtures (committed, in `data/fixtures/`)
3 tracks across 2 genres (2× `dance`, 1× `pop`); the `pop` vector is distant from the `dance`
pair so the switch-genre path is deterministic. Regenerate with
`uv run python scripts/make_fixtures.py`.
