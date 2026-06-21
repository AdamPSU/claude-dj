# ClaudeDJ Recommendation Engine — Implementation Plan

> **Source of truth for autonomous coding agents.** Each phase below is written to be
> executed by an independent agent session with no live dependency on any other phase.
> Read this whole document, then read `contexts/` (project knowledge base) before writing code.

---

## 0. Global rules (apply to EVERY phase — non-negotiable)

### 0.1 Isolation: build inside a new subdirectory
All recommendation-engine code lives in a **new isolated subdirectory** at the repo root:

```
recommendation_engine/
```

Do **not** scatter files into the repo root or into `contexts/`. The only file outside
`recommendation_engine/` you may touch is this plan (to check items off) and `contexts/`
docs (to record corrected assumptions, per `AGENTS.md`).

### 0.2 Package manager: `uv` ONLY
Use [`uv`](https://docs.astral.sh/uv/) for environment and dependency management. Never use
bare `pip`, `poetry`, `conda`, or `requirements.txt` hand-editing.

Bootstrap (run once, by the first agent / Phase 0):
```bash
cd recommendation_engine
uv init --package --name recommendation-engine --python 3.10
```
Add dependencies (per-phase lists are given in each phase):
```bash
uv add requests redis numpy            # example
uv add --dev pytest                    # dev/test deps
```
Run anything through uv so it uses the project venv:
```bash
uv run python -m recommendation_engine.scrape_spotify
uv run pytest
```
`uv` manages `pyproject.toml` + `uv.lock`. Commit both whenever they change.

### 0.3 Commit discipline: Conventional Commits v1.0.0
Commit **frequently** (after each meaningful, working unit — not one giant commit per phase).
Format (https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

- Types used in this project: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `build`, `perf`, `ci`.
- Scope = the phase/module, e.g. `scrape`, `enrich`, `embed`, `store`, `recommend`, `contracts`.
- Breaking change: `!` before the colon **or** a `BREAKING CHANGE:` footer.
- Subject in imperative mood, lower case, no trailing period.
- **Do not add yourself (the agent / Claude) as a co-author.** No `Co-Authored-By:`
  trailer, no "Generated with" footer — commit messages contain only the Conventional
  Commit content.

Examples (use these scopes):
```
chore(engine): scaffold recommendation_engine package with uv
feat(contracts): define artifact schemas and fixtures
feat(scrape): extract track list + ISRC from Spotify playlist
feat(enrich): match tracks to Deezer by ISRC and download previews
feat(enrich): drop tracks whose album has no genre
feat(embed): generate 512-d CLAP music embeddings from previews
feat(store): create Redis vector index and load track hashes
feat(store): compute and store per-genre centroids
feat(recommend): same-genre KNN on positive feedback
feat(recommend): most-distant-genre switch on negative feedback
test(recommend): cover exclusion and ranking with fixtures
docs(engine): document runbook and env vars
```
Branch is already `recommendation_engine`. Do not commit to `main`. Do not push unless asked.

### 0.4 Contracts-first parallelism
Phase 0 produces fixed **artifact contracts** and **fixture files**. Every later phase reads
its input from the contract (real file) OR from the committed fixture, and writes output that
validates against the contract. This is what lets Phases 1–5 be built in parallel without one
phase waiting on another to actually run.

**Validation is mandatory:** each phase ships a validator (a pytest test) that asserts its
output matches the schema in `recommendation_engine/contracts/engine_contracts.md` /
`recommendation_engine/src/recommendation_engine/contracts.py`.

---

## 1. Locked product decisions (do not re-litigate)

| Decision | Value |
| --- | --- |
| Genre source | **Deezer album genre** (`/album/{id}` → `genres.data[0].name`). |
| No-genre policy | If album has no genre, **drop the song before embedding** (no download kept, no vector). |
| Pipeline order | Spotify list → Deezer match → Deezer data + album genre → filter → embed → store → recommend. |
| Positive / neutral feedback | Recommend 5 tracks in the **same genre**, ranked by cosine similarity. |
| Negative feedback | Switch to the **most-distant genre** (largest cosine distance between genre centroids and the current track), recommend 5 there. |
| Ranking | Cosine similarity (Redis KNN score). |
| Embedding model | LAION-CLAP music checkpoint `music_audioset_epoch_15_esc_90.14`, **512-dim, float32, L2-normalized**. |
| Audio source | Deezer 30-second `preview` MP3. **Spotify is used only for the song list**, never for audio/metadata. |
| Deliverable | Plain Python library + scripts (no MCP server here; the harness team wraps it later). |
| Redis index type | `FLAT` (exact KNN) — correct at ~150 tracks. |

**Out of scope here** (belongs to the agent-harness layer, not this engine): the cluster-streak
"stay 3–6 songs" rules from `AGENTS.md`. This engine implements only the same-genre /
most-distant-switch / cosine-ranking behavior above. This is a deliberate omission.

---

## 2. Prerequisites (human setup — blocking only where noted)

1. **Redis** — a Redis Cloud instance with the Search/Vector module is already provisioned
   for this project and the Redis MCP is pointed at it (see repo-root `.mcp.json` / `redis.py`).
   Verified live: `search 8.4.8`, TimeSeries, ReJSON, Bloom. Existing indexes are only
   `memory:*` / `langcache:*`, so `idx:tracks` does not collide.
   - Blocks **Phase 4 & 5** real runs only. Phases 1–3 do not need Redis.
2. **Spotify app** (https://developer.spotify.com/dashboard) — Client ID + Secret + the
   playlist ID. TWO verified constraints (2026-06-20):
   - The app-owner account must have **active personal Premium**, or the API 403s on
     everything (incl. catalog/search). (Premium confirmed working for this project's app.)
   - **Get Playlist Items needs a user token** — the Client-Credentials flow CANNOT read
     `/playlists/{id}/tracks` (always 403). Phase 1 MUST use the **Authorization Code flow**
     with scopes `playlist-read-private playlist-read-collaborative` and a loopback redirect
     URI `http://127.0.0.1:PORT/callback` (Spotify rejects `localhost`). Register that exact
     URI in the app settings. One-time interactive login → store refresh token.
   - Blocks **Phase 1** real run only.
3. **Python 3.14** (repo standard; the engine subdir pins 3.14). Exception: **Phase 3 (CLAP)**
   may need a separate 3.11 venv inside the subdir if `torch`/`laion-clap` lack 3.14 wheels.
   CPU is sufficient for ~150 clips; GPU optional.

Secrets live in `recommendation_engine/.env` (git-ignored). Ship `.env.example`.

---

## 3. Repository layout (target end-state)

```
recommendation_engine/
├── pyproject.toml                # uv-managed
├── uv.lock
├── .env.example
├── .gitignore                    # ignores .env, data/audio/, data/*.json (generated), __pycache__
├── README.md                     # short: how to run each phase end-to-end
├── contracts/
│   └── engine_contracts.md       # human-readable schema spec (Phase 0)
├── data/
│   ├── fixtures/                 # COMMITTED — lets phases run in isolation
│   │   ├── tracks_raw.json
│   │   ├── tracks_enriched.json
│   │   ├── embeddings.jsonl
│   │   └── audio/                # 3 tiny synthetic mp3s for embed tests
│   ├── tracks_raw.json           # generated (git-ignored)
│   ├── tracks_enriched.json      # generated (git-ignored)
│   ├── embeddings.jsonl          # generated (git-ignored)
│   └── audio/                    # generated previews (git-ignored)
├── src/recommendation_engine/
│   ├── __init__.py
│   ├── config.py                 # env loading, constants (DIM=512, paths, rate limits)
│   ├── contracts.py              # dataclasses / validators for artifacts A,B,C + slugify
│   ├── scrape_spotify.py         # Phase 1
│   ├── enrich_deezer.py          # Phase 2
│   ├── embed_clap.py             # Phase 3
│   ├── store_redis.py            # Phase 4
│   └── recommend.py              # Phase 5
└── tests/
    ├── test_contracts.py
    ├── test_scrape.py
    ├── test_enrich.py
    ├── test_embed.py
    ├── test_store.py
    └── test_recommend.py
```

---

## 4. Phase 0 — Contracts & fixtures (DO FIRST)

**Goal:** produce the shared contract + fixtures + package scaffold so Phases 1–5 fork in parallel.
**Owner:** one agent. **Blocks:** everything. **Needs:** nothing external.

### 4.1 Tasks
1. Scaffold the package with `uv` (§0.2) and create the directory layout (§3).
2. Add `.gitignore` and `.env.example`.
3. Write `contracts/engine_contracts.md` (schemas below) and `src/recommendation_engine/contracts.py`.
4. Write the committed fixtures under `data/fixtures/`.
5. Implement and test `slugify_genre`.

### 4.2 Canonical identifiers
- `deezer_id`: string of Deezer's numeric track id.
- `id`: `f"deezer:{deezer_id}"`.
- Redis key: `track:{deezer_id}`. Centroid key: `genre_centroid:{genre_tag}`.

### 4.3 `slugify_genre` (used everywhere genre is a filter/key)
Deezer genres contain `/` and spaces (e.g. `"Rap/Hip Hop"`) which break RediSearch TAG queries.
Store a **display** value and a **slug**:
```python
import re
def slugify_genre(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)   # "Rap/Hip Hop" -> "rap_hip_hop"
    return s.strip("_")
```
- `genre`     = original display string (e.g. `"Rap/Hip Hop"`).
- `genre_tag` = `slugify_genre(genre)` (e.g. `"rap_hip_hop"`). **This is the indexed TAG.**

### 4.4 Artifact A — `data/tracks_raw.json` (Phase 1 → Phase 2)
JSON array; each item:
```json
{
  "spotify_id": "string",
  "title": "string",
  "artist": "string",
  "isrc": "string (uppercase, may be empty if unknown)",
  "album_name": "string"
}
```

### 4.5 Artifact B — `data/tracks_enriched.json` + `data/audio/{deezer_id}.mp3` (Phase 2 → 3,4)
JSON array; each item (one **surviving** track, i.e. has a genre + downloaded preview):
```json
{
  "id": "deezer:3135556",
  "deezer_id": "3135556",
  "spotify_id": "string",
  "isrc": "string",
  "title": "string",
  "artist": "string",
  "album": "string",
  "genre": "Dance",
  "genre_tag": "dance",
  "artwork_url": "string",
  "preview_source": "deezer",
  "duration_seconds": 226,
  "rank": 814839,
  "mp3_path": "data/audio/3135556.mp3",
  "clip_hash": "sha256-hex",
  "match_method": "isrc | search"
}
```

### 4.6 Artifact C — `data/embeddings.jsonl` (Phase 3 → Phase 4)
One JSON object per line:
```json
{"id": "deezer:3135556", "vector": [/* exactly 512 floats, L2-normalized */]}
```

### 4.7 Redis schema (Phase 4 → Phase 5)
- Hash `track:{deezer_id}` fields (all strings except `embedding`):
  `title, artist, album, genre, genre_tag, isrc, deezer_id, spotify_id, artwork_url,
   duration_seconds, rank, embedding`
  where `embedding` = `numpy.array(vector, dtype=numpy.float32).tobytes()`.
- Index `idx:tracks`:
  ```
  FT.CREATE idx:tracks ON HASH PREFIX 1 track: SCHEMA
    genre_tag TAG
    artist    TAG
    title     TEXT
    embedding VECTOR FLAT 6 TYPE FLOAT32 DIM 512 DISTANCE_METRIC COSINE
  ```
- Centroid `genre_centroid:{genre_tag}` hash fields: `genre_tag`, `count`,
  `embedding` (512×float32 bytes, L2-normalized mean of the genre's vectors).

### 4.8 Recommendation API (Phase 5 surface)
```python
def next_five(current_track_id: str,
              signal: str,                       # "positive" | "neutral" | "negative"
              recently_played: list[str] = None  # ids to exclude
              ) -> list[str]:                     # up to 5 track ids, best first
    ...
```

### 4.9 Fixtures (committed)
- `tracks_raw.json`: 3 realistic rows (with valid-looking ISRCs).
- `tracks_enriched.json`: 3 rows spanning **2 distinct genres** (e.g. 2× `dance`, 1× `pop`)
  so recommend tests have both same-genre and switch-genre paths.
- `audio/`: 3 tiny synthetic MP3s (generate 1–2 s sine tones via `numpy` + `soundfile`,
  or use any short public-domain clip). These exist so Phase 3 can run without Deezer.
- `embeddings.jsonl`: 3 lines of 512-float L2-normalized vectors. To make the switch-genre
  test meaningful, make the `pop` vector clearly distant from the two `dance` vectors.

### 4.10 Acceptance
- `uv run pytest tests/test_contracts.py` passes: every fixture validates against `contracts.py`;
  `slugify_genre("Rap/Hip Hop") == "rap_hip_hop"`.

### 4.11 Suggested commits
```
chore(engine): scaffold recommendation_engine package with uv
feat(contracts): define artifact A/B/C schemas and slugify_genre
test(contracts): validate committed fixtures against schemas
```

---

## 5. Phase 1 — Spotify playlist → song list (`scrape_spotify.py`)

**Input:** Spotify creds + playlist ID. **Output:** Artifact A. **Needs Redis?** No.
**Deps:** `uv add requests`.

> **Auth flow = Authorization Code (user OAuth), NOT Client-Credentials.** Verified
> 2026-06-20: Client-Credentials returns 403 on `/playlists/{id}/tracks` (no user context).
> A user token is required. The app owner must also have active Premium (see §2).

### 5.1 One-time OAuth setup (`spotify_auth.py` helper)
1. Register redirect URI `http://127.0.0.1:8888/callback` in the Spotify app settings
   (Spotify rejects `localhost` — use the loopback IP).
2. Build the consent URL: `GET https://accounts.spotify.com/authorize?response_type=code`
   `&client_id=...&redirect_uri=http://127.0.0.1:8888/callback`
   `&scope=playlist-read-private%20playlist-read-collaborative`.
3. Run a tiny local HTTP server on `127.0.0.1:8888` to capture the `?code=...` redirect;
   open the consent URL in the browser; user logs in and approves.
4. Exchange the code: `POST /api/token` with `grant_type=authorization_code`, the `code`,
   `redirect_uri`, and Basic-auth `client_id:client_secret`. Persist the **refresh token**
   to `.env` (or `data/.spotify_token.json`, git-ignored).
5. This is interactive — run it via the session `! uv run python -m recommendation_engine.spotify_auth`
   so the browser can open.

### 5.2 Steps (each run)
1. Load the stored refresh token; get a fresh access token:
   `POST /api/token` with `grant_type=refresh_token` + Basic auth. (No browser needed.)
2. Normalize the playlist id: strip any `?si=`/URL wrapper (e.g. `2tZuU4...?si=abc` → `2tZuU4...`).
3. Page through: `GET https://api.spotify.com/v1/playlists/{playlist_id}/tracks?limit=100`
   following `next` until null. Use `fields=next,items(track(id,name,artists(name),external_ids(isrc),album(name)))`.
4. For each item (`item.track`):
   - `title`=`track.name`, `artist`=`track.artists[0].name`, `spotify_id`=`track.id`,
     `isrc`=`track.external_ids.isrc` (uppercase; `""` if missing), `album_name`=`track.album.name`.
   - Skip `null`/non-`track` items (local/unavailable/episodes). Log skipped + empty-ISRC counts.
5. Write Artifact A to `data/tracks_raw.json`.

### 5.3 Pitfalls
- Access token expires (~1h) — refresh via the stored refresh token; handle 401 by refreshing once.
- Editorial/algorithmic playlists (`37i9...`) now 404 — only normal user playlists work.
- `GET /playlists/{id}` returns 200 with `tracks: null` even when items are unreadable — don't
  rely on it; the `/tracks` endpoint is the source of truth.
- Validate the output with the Artifact A validator before exit.

### 5.4 Acceptance
- Against the real playlist, produces a non-empty valid `tracks_raw.json`.
- `uv run pytest tests/test_scrape.py` (mock the HTTP layer with the expected JSON shape).

### 5.5 Suggested commits
```
feat(scrape): add Spotify Authorization Code OAuth helper
feat(scrape): refresh-token auth and playlist-id normalization
feat(scrape): paginate playlist items and extract title/artist/isrc/album
test(scrape): cover pagination and null-track skipping
```

---

## 6. Phase 2 — Deezer enrichment + preview download + genre filter (`enrich_deezer.py`)

**Input:** Artifact A (real or `data/fixtures/tracks_raw.json`). **Output:** Artifact B + MP3s.
**Needs Redis?** No. **Deps:** `uv add requests` (numpy/soundfile only if generating fixtures).

Deezer public API, no auth. Base: `https://api.deezer.com`.

### 6.1 Steps (per track)
1. **Match to a Deezer track:**
   - Primary (if ISRC present): `GET /track/isrc:{isrc}`. Success → use returned track object;
     `match_method = "isrc"`.
   - Fallback: `GET /search?q=artist:"{artist}" track:"{title}"` → take `data[0]`;
     `match_method = "search"`. If no result, **drop** (log).
2. From the matched track object read: `id` (→ `deezer_id`), `duration`, `rank`, `preview`
   (30 s MP3 URL), `md5_image`/`album.cover_xl` (→ `artwork_url`), `album.id`, `album.title`,
   `artist.name`. If `preview` is empty, **drop** (log).
3. **Genre (mandatory):** `GET /album/{album.id}` → `genres.data`.
   - If `genres.data` is empty/absent → **DROP the song. Do not download. Do not embed.** Log it.
   - Else `genre = genres.data[0].name`; `genre_tag = slugify_genre(genre)`.
   - **Cache album responses** by `album.id` (many tracks share an album) to cut requests.
4. **Download preview:** GET the `preview` URL → save bytes to `data/audio/{deezer_id}.mp3`.
   Compute `clip_hash = sha256(bytes).hexdigest()`.
5. Emit the Artifact B row.

### 6.2 Rate limiting & robustness
- Deezer limits ~**50 requests / 5 seconds**. Add a token-bucket / sleep limiter and dedupe
  album fetches. Retry transient errors (429/5xx) with backoff.
- Deezer error payloads come back as `{"error": {...}}` with HTTP 200 — check for an `error` key.
- Preview URLs expire — that's why we download now and persist `clip_hash` + `deezer_id`,
  **never** the URL as source of truth.

### 6.3 Acceptance
- Every output row has a non-empty `genre`/`genre_tag` and an existing `mp3_path` file.
- Drop reasons (no-match / no-preview / no-genre) are logged with counts.
- `uv run pytest tests/test_enrich.py` (mock HTTP; assert no-genre rows are dropped and
  album responses are cached / fetched once per album id).

### 6.4 Suggested commits
```
feat(enrich): match tracks to Deezer by ISRC with search fallback
feat(enrich): fetch album genre and drop genre-less tracks
feat(enrich): download previews and record sha256 clip hash
feat(enrich): rate-limit and cache Deezer album lookups
test(enrich): cover drop rules and album cache
```

---

## 7. Phase 3 — CLAP embeddings (`embed_clap.py`)

**Input:** Artifact B + MP3s (real or fixtures). **Output:** Artifact C. **Needs Redis?** No.
**Deps:** `uv add laion-clap librosa numpy` (CLAP pulls `torch`/`torchaudio`/`transformers`).

### 7.1 Steps
1. Load the **music** checkpoint once:
   ```python
   import laion_clap
   model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
   model.load_ckpt("music_audioset_epoch_15_esc_90.14.pt")  # download from LAION-CLAP releases
   ```
   (If the `.pt` isn't present, fetch it from the LAION-CLAP model release and store its path
   in `config.py`. Verify the checkpoint matches `amodel`/512-dim before embedding.)
2. For each Artifact B row, load `mp3_path`, resample to **48 kHz mono** (`librosa.load(path, sr=48000, mono=True)`).
3. Embed: `model.get_audio_embedding_from_filelist(x=[paths], use_tensor=False)` (batch in chunks),
   yielding a 512-vector per file.
4. **L2-normalize** each vector (`v / np.linalg.norm(v)`); assert length 512 and all finite.
5. Append `{"id": ..., "vector": [...]}` to `data/embeddings.jsonl`.
6. Skip + log any unreadable/too-short clip.

### 7.2 Pitfalls
- Checkpoint/`amodel` mismatch → wrong dim or garbage. Pin both.
- Keep batches small to bound memory; CPU is fine for ~150 clips.
- Optionally discard `data/audio/` after this phase (provenance survives via `clip_hash`).

### 7.3 Acceptance
- Running on `data/fixtures/` produces 3 lines, each a finite 512-vector with norm ≈ 1.0.
- `uv run pytest tests/test_embed.py` (asserts dim=512, unit norm, id coverage).

### 7.4 Suggested commits
```
feat(embed): load LAION-CLAP music checkpoint
feat(embed): embed 48kHz previews to L2-normalized 512-d vectors
test(embed): assert dimension, unit norm, and id coverage
```

---

## 8. Phase 4 — Redis index + store + centroids (`store_redis.py`)

**Input:** Artifacts B + C (real or fixtures) + live Redis. **Output:** populated Redis.
**Needs Redis?** Yes (Prereq 1). **Deps:** `uv add redis numpy`.

> Product code uses **redis-py** (`FT.CREATE` / `FT.SEARCH` via `redis.Redis`). The Redis **MCP**
> is for the agent to *inspect/verify* during development (`get_indexes`, `dbsize`,
> `vector_search_hash`), not a runtime dependency of the shipped code.

### 8.1 Steps
1. Connect with `redis.Redis(host, port, decode_responses=False)` (bytes mode — vectors are binary).
2. Create the index if absent (idempotent: catch "Index already exists"):
   ```
   FT.CREATE idx:tracks ON HASH PREFIX 1 track: SCHEMA
     genre_tag TAG  artist TAG  title TEXT
     embedding VECTOR FLAT 6 TYPE FLOAT32 DIM 512 DISTANCE_METRIC COSINE
   ```
3. Join B (metadata) + C (vector) by `id`. For each track:
   `HSET track:{deezer_id}` with all metadata fields (as UTF-8 bytes/strings) and
   `embedding = np.asarray(vector, dtype=np.float32).tobytes()`.
4. **Centroids:** group vectors by `genre_tag`, compute the mean vector, L2-normalize, and
   `HSET genre_centroid:{genre_tag}` with `embedding` (float32 bytes), `count`, `genre_tag`.
5. Print a summary: tracks loaded, genres seen, centroids written.

### 8.2 Pitfalls
- `embedding` MUST be float32 little-endian bytes of length `512*4 = 2048`; mismatched dtype/dim
  silently corrupts search. Add an assertion.
- TAG values must be the **slug** (`genre_tag`), never the raw genre (avoids `/`/space escaping).
- Make load idempotent (re-running overwrites the same keys cleanly).

### 8.3 Acceptance
- After load, indexed key count == number of input tracks (verify via MCP `get_indexed_keys_number`
  or `FT.INFO idx:tracks`).
- A self-KNN of any stored vector returns that same track with distance ≈ 0.
- `uv run pytest tests/test_store.py` against a local Redis (skip/xfail if unreachable).

### 8.4 Suggested commits
```
feat(store): create idx:tracks FLAT cosine vector index
feat(store): load track hashes with float32 embeddings
feat(store): compute and persist per-genre centroids
test(store): verify index count and self-KNN distance
```

---

## 9. Phase 5 — Recommendation engine (`recommend.py`)

**Input:** populated Redis (or fixtures loaded into a scratch Redis). **Output:** `next_five(...)`.
**Needs Redis?** Yes. **Deps:** `uv add redis numpy`.

### 9.1 `next_five(current_track_id, signal, recently_played=None)`
1. Load current track hash → its `embedding` (bytes → np.float32) and `genre_tag`.
2. Decide the **target genre_tag**:
   - `signal in {"positive", "neutral"}` → target = current `genre_tag`.
   - `signal == "negative"` → load all `genre_centroid:*`; compute cosine distance between the
     current vector and each centroid; **target = genre_tag with the LARGEST distance**, excluding
     the current genre. (If only one genre exists, fall back to same genre and log.)
3. KNN query, filtered by the target tag, asking for `K = 5 + len(exclusions) + buffer`:
   ```
   FT.SEARCH idx:tracks "(@genre_tag:{<target>})=>[KNN <K> @embedding $vec AS score]"
     PARAMS 2 vec <current_vector_bytes>
     SORTBY score ASC  RETURN 1 score  DIALECT 2
   ```
   (`score` = cosine distance; lower = more similar.)
4. Build the result: drop `current_track_id` and any id in `recently_played`; keep order
   (ascending distance = descending similarity); **return the first 5 ids**.
5. Tie-break deterministically: if scores are equal, order by `rank` then `id`.
6. Optional stretch: cap to ≤2 tracks per artist before truncating to 5.

### 9.2 Pitfalls
- Always over-fetch (`K > 5`) then filter exclusions, so you can still return 5.
- Pass the query vector as float32 bytes in `PARAMS`; use `DIALECT 2`.
- Negative path needs ≥2 genres in the corpus to be meaningful — the curated corpus (§ contexts)
  should span multiple Deezer genres.

### 9.3 Acceptance
- **Positive:** returns 5 same-genre ids ranked by similarity; `current_track_id` never appears.
- **Negative:** returns 5 ids whose `genre_tag` ≠ current and == the most-distant genre.
- **Exclusion:** ids in `recently_played` never appear.
- `uv run pytest tests/test_recommend.py` using the committed fixtures loaded into a scratch
  Redis (the `pop` vs `dance` separation in fixtures makes the switch path deterministic).

### 9.4 Suggested commits
```
feat(recommend): load current vector and resolve target genre
feat(recommend): same-genre KNN ranking for positive feedback
feat(recommend): most-distant-genre switch for negative feedback
feat(recommend): apply recent-played exclusion and deterministic tie-break
test(recommend): cover positive, negative, and exclusion paths
```

---

## 10. Integration runbook (after parallel phases land)

Run once for real data, in order, from `recommendation_engine/`:
```bash
cp .env.example .env            # fill in Spotify creds + playlist id + Redis url
uv run python -m recommendation_engine.scrape_spotify     # -> data/tracks_raw.json
uv run python -m recommendation_engine.enrich_deezer      # -> data/tracks_enriched.json + audio/
uv run python -m recommendation_engine.embed_clap         # -> data/embeddings.jsonl
uv run python -m recommendation_engine.store_redis        # -> Redis (idx:tracks + centroids)
uv run pytest                                             # full suite
```
Then smoke-test recommendations:
```python
from recommendation_engine.recommend import next_five
next_five("deezer:3135556", "positive")
next_five("deezer:3135556", "negative")
```
Final integration commit:
```
docs(engine): add runbook and .env.example
chore(engine): end-to-end dry run on demo corpus
```

---

## 11. Definition of done
- [ ] All code under `recommendation_engine/`, nothing leaked to repo root/`contexts/`.
- [ ] `uv.lock` committed; `uv run pytest` green.
- [ ] Phases 1–5 each validate their artifact against `contracts.py`.
- [ ] Genre-less songs are provably dropped before embedding (test covers it).
- [ ] Positive → same-genre 5; Negative → most-distant-genre 5; exclusions honored.
- [ ] Commits follow Conventional Commits; no commits on `main`.
- [ ] Any corrected API assumption recorded in `contexts/` per `AGENTS.md`.
