# Recommendation
Use Deezer API as the easiest source for the embedding corpus.

Why:
- It returns real 30-second MP3 preview URLs directly in track/search/chart responses.
- It does not require OAuth for basic public search/chart calls.
- Track objects include useful metadata: id, title, artist, album, artwork, isrc, rank, duration, preview.
- isrc lets you later match the same recording to Spotify for playback/display.
- Spotify’s preview_url is deprecated/nullable and Spotify explicitly restricts using Spotify content for AI/ML ingestion, so do not build embeddings from Spotify audio.

# Spotify access constraint (verified 2026-06-20)
Spotify's Feb 2026 policy change requires the account that OWNS the developer app to have
an active **Premium** subscription. Without it, the Web API returns `403 "Active premium
subscription required for the owner of the app"` on ALL endpoints — including basic catalog
reads and search — under the Client-Credentials flow. The token still issues (200); only data
calls fail. Notes:
- Subscription changes take a few hours to propagate.
- Personal Premium works; Family-plan *master* accounts reportedly still 403.
- No code workaround — it is account/app-level.
- Mitigation if Premium is unavailable: Spotify is only the seed-list source, so Phase 1 can
  instead seed from a Deezer playlist/chart (no auth) or a manual artist/title/ISRC list.
- Decision (2026-06-20): keep Spotify, own the app under a personal Premium account.

Second constraint (verified 2026-06-20): **Get Playlist Items requires a user token.**
After the Premium gate is cleared (catalog `search` returns 200), `GET /playlists/{id}` still
reads (200) but `GET /playlists/{id}/tracks` returns a generic `403 Forbidden`, and the meta
object's `tracks` field is `null`. Per Spotify docs, the Client-Credentials flow has no user
context, and playlist items are only readable by the playlist owner/collaborator — so Phase 1
MUST use the **Authorization Code flow** (user OAuth), not Client-Credentials.
- Scopes: `playlist-read-private playlist-read-collaborative`.
- Redirect URI must be loopback `http://127.0.0.1:PORT/callback` (Spotify disallows `localhost`).
- Flow: one-time interactive login -> store the refresh token -> Phase 1 refreshes silently.
- Editorial/algorithmic playlists (37i9... ids) now return 404 on the API regardless of auth.
  The target playlist must be a normal user playlist (the user owns "asleep among endives").

# API Options

| Source | Fit | Notes |
| -------- | -------- | -------- |
|Deezer  |  Best default  |  Easy 30-second previews, searchable catalog, useful metadata, ISRC matching. |
| iTunes Search API  |  Good fallback  |  Has previewUrl, no auth, but terms are more promotional/streaming-oriented. |
|Jamendo |   Safest legal fallback  |  Creative Commons/full audio, but less recognizable mainstream catalog. |
|Spotify  |  Metadata/playback only |   Use for playlists, playback, artwork, IDs. Do not rely on preview audio. |

# Easiest Path
Build a one-time corpus from Deezer, then enrich/match to Spotify.
1. Pick 6-8 demo “music pockets.”
2. Pull 25-40 Deezer candidates per pocket using search/chart/radio/playlist endpoints.
3. Filter to 100-200 strong tracks:
- has preview
- has isrc
- readable/playable
- duration roughly 2-6 minutes
- avoid too many tracks by the same artist
- prefer high rank
4. Download the preview only during prep.
5. Run CLAP audio embedding on the 30-second clip.
6. Store the vector and metadata in Redis/Redis AI.
7. Discard raw audio.
8. Optionally resolve each track to Spotify by ISRC for playback.

# Suggested Demo Corpus
Build around adjacent clusters, not random hits. The DJ needs smooth transitions.
A good 160-track set:
- Pocket
- Reggaeton / Latin pop
- Dancehall / dembow
- Afrobeats / amapiano
- House / disco house
- Synthwave / electronic pop
- Indie pop / alt pop
- Lo-fi / chillhop / downtempo
- Hip-hop / trap / rhythmic pop

This gives the agent enough density to “stay close” for 3-6 tracks and enough adjacency to shift naturally.

# Metadata To Store
For each track:
```json
{
  "id": "deezer:3135556",
  "deezer_id": "3135556",
  "spotify_id": "optional-after-isrc-match",
  "isrc": "GBDUW0000059",
  "title": "Harder, Better, Faster, Stronger",
  "artist": "Daft Punk",
  "album": "Discovery",
  "artwork_url": "...",
  "preview_source": "deezer",
  "source_query": "electronic dance",
  "demo_pocket": "house_electronic",
  "duration_seconds": 226,
  "rank": 814839,
  "embedding_model": "CLAP:<version>",
  "embedding_input": "30s_deezer_preview",
  "clip_hash": "...",
  "cluster_id": "electronic_a"
}
```

Do not persist expiring preview URLs as the source of truth. Store Deezer ID, ISRC, provenance, and clip hash.

# DJ Database Quality
The database should optimize for transitions, not coverage.
Rules I’d use:
- At least 15-25 tracks per cluster.
- Every cluster should have 1-2 adjacent clusters.
- Include bridge genres like Latin pop, dancehall, afrobeats, disco house, synth pop.
- Avoid isolated novelty tracks.
- Cap each artist at 3-5 tracks.
- Add manual tags for demo controllability: energy, danceable, chill, vocals, dark, bright.

# Final Path
Use Deezer previews for CLAP embeddings, Spotify for playback/display where possible, and Redis vector search as the recommendation layer. For a hackathon demo, 150 well-curated Deezer preview embeddings is enough to make the DJ feel intentional.

# Implementation findings (verified 2026-06-20, recommendation_engine build)

## Python 3.14 + CLAP stack — wheels exist
The engine pins Python 3.14. The earlier worry that `torch`/`laion-clap`/`librosa` lack
3.14 wheels did NOT materialize. `uv add laion-clap librosa` resolves on 3.14 and pulls
cp314 wheels: `laion-clap 1.1.7`, `librosa 0.11.0`, `torch 2.12.1`, `torchvision 0.27.1`
(`numpy` pinned to `1.26.4` by numba/librosa, still satisfies `>=1.26`). No separate 3.11
venv needed. Heavy imports are still kept lazy in `embed_clap.py` so the module imports
without the torch stack present. The CLAP music checkpoint itself is not bundled — set
`CLAP_CHECKPOINT` to a local `.pt` path (downloads fail in sandboxes: SSL verify error).

## redis-py 8.0 high-level FT search is broken against Redis 8 (RESP3)
`client.ft(index).search(Query(...))` returns **zero docs** against the Redis Cloud /
Redis 8 RESP3 reply even when the index is populated (`num_docs` is correct, raw search
returns rows). Fix: issue the raw `FT.SEARCH ... DIALECT 2` via `execute_command` and parse
the reply yourself. In bytes mode (`decode_responses=False`, required for binary vectors)
the RESP3 reply is a **dict with bytes keys** (`b'total_results'`, `b'results'`, each row a
dict with `b'id'` and `b'extra_attributes'`); handle both bytes and str keys, and the RESP2
flat-list form. Both `recommend.py` and `tests/test_store.py` use this raw-parse approach.
Also note: `IndexDefinition`/`IndexType` import from `redis.commands.search.index_definition`
(snake_case) in redis-py 8.

## Redis connection
Redis Cloud uses ACL; the project `.env` sets `REDIS_HOST/PORT/USERNAME/PASSWORD`
(`REDIS_USERNAME=default`). Password-only AUTH works for the `default` user, but both Redis
clients pass `username` too so a non-default ACL user also works.

# Live end-to-end run findings (verified 2026-06-20, full pipeline on real playlist)

Ran scrape -> enrich -> embed -> store against a real 231-track playlist. Key gotchas:

## Spotify: /playlists/{id}/tracks is dead for Development-mode apps
- `GET /v1/playlists/{id}/tracks` now returns **403 Forbidden** (and playlist objects
  come back with the `tracks` paging object stripped) for apps in **Development mode**.
- The working replacement is **`GET /v1/playlists/{id}/items`**. Same paging (`next`,
  `total`), BUT each row's track object is keyed **`item`** (not `track`). Note the
  confusing `"track": true` boolean *inside* `item`. `fields` becomes
  `next,items(item(id,name,artists(name),external_ids(isrc),album(name)))`.
- `scrape_spotify.py` uses `/items` + the `item` key accordingly.
- Individual `GET /tracks/{id}`, `/albums/{id}`, and `/search` still work fine in dev mode.

## Spotify OAuth: HTTPS redirect required, even for loopback
- The dashboard now refuses `http://` redirect URIs entirely — even `http://127.0.0.1`.
  Must register `https://127.0.0.1:8888/callback`.
- Our plain-HTTP loopback catcher therefore fails. Fix: run the local callback server
  over **HTTPS with a self-signed cert** (openssl one-liner, `ssl.wrap_socket`). The
  browser shows a one-time "not private" warning -> Advanced/Proceed. See
  `recommendation_engine/authorize_and_save.py` (mints + writes + self-tests the token).
- Spotify access tokens start `BQ` (~280 chars); refresh tokens start `AQ` (~130 chars).
  A 400 `invalid_grant: Invalid refresh token` means the value is wrong/revoked.

## Spotify: Development-mode user allowlist
- Dev-mode apps only work for users explicitly added under **Dashboard -> User Management**
  (up to 25). A non-allowlisted account that authorizes gets a valid token but every API
  call returns **403 "The user is not registered for this application."** Add the correct
  account (by email) AND authorize with that same account.

## Deezer: preview download can time out -> must not crash the run
- `cdnt-preview.dzcdn.net` occasionally `ReadTimeout`s. The original `download()` had no
  network-exception handling, so one stalled CDN response killed the whole loop after
  writing nothing. Fixed: retry on `requests.exceptions.RequestException` in both
  `get_json` and `download`, and wrap the per-track call in `enrich_tracks` so a single
  failure drops that track (`dropped_error`) instead of aborting. 229/231 enriched.

## CLAP checkpoint
- If `music_audioset_epoch_15_esc_90.14.pt` is present locally, pass it via
  `CLAP_CHECKPOINT` or `embed_clap --checkpoint <path>`. If no local checkpoint is found,
  `laion-clap` downloads/uses its package default `630k-audioset-best.pt`, which matches
  `HTSAT-tiny`. Do not pair the package default checkpoint with `HTSAT-base`; live import
  validation showed that combination fails with state-dict size mismatches.
- CPU embedding of ~230 clips takes a couple of minutes.

# Live import-history findings (verified 2026-06-21)

- `uv run python -m recommendation_engine.import_history` successfully read Spotify
  recently played history, matched the track through Deezer, embedded the preview with
  CLAP, stored the imported Redis track, and published `claudedj:initial_seed_track_id`.
- Redis Cloud timed out through redis-py RESP3 on binary `HSET` during the live import.
  The import-history path now uses a minimal raw RESP2 client for the binary track write,
  TTL, seed pointer, and recommendation check, matching the backend recommendation bridge.
- The backend harness must hydrate/register the imported seed track after Redis search.
  Claude may include `initial_seed_track_id` in the startup queue, so the app-owned catalog
  needs the seed track itself, not just the returned neighbors.
