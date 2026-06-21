# DJ agent technical design

## Architecture

The system has four layers:

- Claude Code SDK: runs the DJ agent with a custom mission prompt.
- MCP server: exposes tools Claude can call.
- Redis: stores vectors, state, memory, event streams, and reaction traces.
- Desktop mascot app: shows the ClaudeDJ mascot first, with current song and compact DJ status layered in later.

Claude is the high-level queue manager. Redis is the memory and retrieval layer. The MCP server is the boundary between the agent and the music system.

## Agent harness

Claude Code SDK runs with a DJ-specific system prompt:

- On startup, choose an initial 2-4 song demo set; do not extend the queue beyond that set immediately.
- Search embeddings before narration.
- Narrate before starting playback and when changing direction.
- React to thresholded reaction/cluster-policy events rather than polling Claude mid-song for neutral checks.
- If positive, keep the current set going.
- If negative and changing genres/clusters, narrate the bridge before replacing the upcoming set.
- Do not wait until the song ends to decide what comes next.
- Keep user-facing narration short.

The agent should act through MCP tools only. It should not directly modify Redis or playback internals.

Current implementation note: `src/backend/claude_dj` now has a Python Claude Agent SDK harness. `claude_dj/main.py` is the script entry module. `agent/` owns the SDK client, runner, and hook prompt loading; `mcp/` owns the project DJ MCP tool handlers, in-process SDK MCP server, and Deepgram-backed narration implementation; `transition.py` owns deterministic track-boundary transition execution. Prompt text lives in `agent/prompts/*.md` with YAML frontmatter and simple XML sections.

Sentry observability is configured for both app surfaces: the Next.js App Router frontend uses `@sentry/nextjs` browser, server, and edge config files; the Python backend initializes `sentry-sdk` before FastAPI app creation and when the autonomous CLI harness starts. The CLI module must stay import-side-effect free so backend unit tests can import helpers without emitting Sentry telemetry; CI backend tests do not receive the runtime `SENTRY_DSN`. The authenticated Sentry MCP account sees org `pennsylvania-state-universi-og` and project `javascript`; `src/frontend/.env.example` and `src/backend/.env.example` include the public DSN and expected env keys. Configure `SENTRY_AUTH_TOKEN` only in CI or local build environments that upload source maps.

Sentry alerts are configured for demo-critical backend and frontend issues. Backend events are tagged `service=claude_dj_backend` and include spans/breadcrumbs for Claude SDK turns, MCP tool calls, Deepgram narration, and track-boundary transitions. Frontend events are tagged `service=claude_dj_frontend` in browser, server, and edge Sentry config.

Pipeline agents can tag backend Sentry spans/errors with `CLAUDE_DJ_COLLABORATION_ID`, `CLAUDE_DJ_AGENT_ID`, `CLAUDE_DJ_AGENT_NAME`, `CLAUDE_DJ_WORKSTREAM`, `CLAUDE_DJ_SCENARIO`, `CLAUDE_DJ_TASK_KIND`, and `CLAUDE_DJ_VERIFICATION_ID`. Use distinct agent names for parallel validation and keep values pseudonymous. The Sentry dashboard `ClaudeDJ Observability` is dashboard id `7339119` and includes widgets for recent ClaudeDJ runs, MCP tool calls over time/by tool, backend/frontend error trends, and backend/frontend error groups. `python -m claude_dj.reporting.sentry_report --input report.json --output report.pdf` writes a compact PDF from collected Sentry query summaries and agent findings without adding PDF dependencies.

## MCP tools

Minimum useful tool set:

- `search_track_embeddings`: search Redis vectors by text query, seed track, similarity, or shift mode.
- `get_current_playback`: return current track, progress, seconds remaining, and queue.
- `get_reaction_signal`: return current and recent engagement scores.
- `replace_queue`: replace upcoming tracks with selected candidates.
- `play_track`: start playback from a selected track.
- `narrate`: speak and/or display a short DJ line. Spoken narration should use Deepgram TTS when available.
- `mark_track_feedback`: persist liked, disliked, or neutral feedback for a track/cluster.
- `get_session_context`: return compact context for the next DJ decision.
- `summarize_session`: write a compact end-of-session memory.
- `search_session_history`: search previous listening sessions by time range, track, artist, cluster, reaction, or semantic query.

Nice-to-have tools:

- `get_cluster_streak`
- `set_cluster_policy`
- `get_yesterday_summary`
- `record_demo_event`

## Repo MCP client configuration

The repo includes developer/client MCP config for Claude Code and OpenCode:

- `sentry`: remote Sentry MCP over HTTP at `https://mcp.sentry.dev/mcp`; `.mcp.json` configures Claude-style clients and `opencode.json` configures OpenCode with OAuth and a 120s timeout. Authenticate each client through its own MCP OAuth flow.
- `redis`: official Redis MCP via `uvx --from redis-mcp-server@latest redis-mcp-server --url redis://default:${REDIS_PASSWORD}@sugar-daylit-corn-40583.db.redis.io:18497/0`; provide `REDIS_PASSWORD` in the client environment.
- `deepgram`: Deepgram CLI MCP via `uvx --from deepctl dg mcp --non-interactive`; authenticate with `dg login` or provide `DEEPGRAM_API_KEY` in the environment.

These are developer/client MCPs. The product's custom DJ MCP server remains the runtime tool boundary for playback, retrieval, memory, and narration.

## Narration audio

Use Deepgram for generated DJ narration audio, likely through Aura Text-to-Speech. The runtime `narrate` tool should keep text short. Current implementation keeps generated audio short-lived in memory by narration id; there is no persistent narration cache.

The `immediate` narration mode is used for startup narration before playback begins. The `prepare` narration mode is used during event-driven shift preparation, generates audio before the current track ends, and stores a ready transition plan so the boundary path does not call Deepgram.

Live smoke test note: `aura-2-thalia-en` generated valid audio through `/v1/speak`; the response returned `audio/mpeg` bytes. The runtime preserves Deepgram's returned `content_type` instead of assuming a fixed container.

Voice direction: prefer a confident personal music-guide persona, modeled after Spotify DJ's public positioning rather than copied by name. The DJ should feel like a human, paced, conversational guide that gives short context around a musical pocket, balances nostalgia and discovery, and responds to the moment without turning into a hype man. Deepgram Aura voices are selected by model identifiers such as `[modelname]-[voicename]-[language]`; Deepgram docs do not label race or ethnicity, so choose from documented voice traits rather than inferred identity. Current default is `aura-2-luna-en` at speed `1.3`. Deepgram does not currently expose an emotion/style knob for Aura-2 REST TTS; excitement should come from voice choice, speed, and concise DJ-style copy. Continue auditioning voices before the final demo.

## Spotify playback

ClaudeDJ keeps an app-owned queue instead of relying on Spotify's native queue. Spotify's Web API can start/resume playback and add to the native queue, but it does not provide a reliable clear-and-replace queue primitive for the user's player. Therefore `replace_queue` updates ClaudeDJ's active or pending queue, `play_track` starts the selected Spotify URI, and `get_current_playback` reconciles Spotify's current player state with the app-owned queue.

Runtime Spotify credentials are `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, and `SPOTIFY_REFRESH_TOKEN`. The refresh token needs `user-read-playback-state`, `user-modify-playback-state`, `streaming`, `playlist-read-private`, and `playlist-read-collaborative`. The `streaming` scope is required for the Web Playback SDK; the playback-state scopes cover Web API device/playback control. ClaudeDJ omits Spotify's optional `device_id`, so playback targets the user's currently active Spotify device. `CLAUDE_DJ_DEMO_TRACK_URIS` can provide comma-separated Spotify track URIs for the fixture catalog if Spotify playlist/search is unavailable.

Spotify refresh tokens are procured through Authorization Code flow, not from the developer dashboard. Current Spotify redirect URI docs require an explicit local loopback IP for local OAuth redirects, such as `http://127.0.0.1:8888/callback`; do not use `localhost`. The redirect URI in the dashboard, authorize URL, and token exchange must match exactly. After approval, exchange only the `code` query parameter for tokens; do not store the full redirected query string or `ubi` parameter as `SPOTIFY_REFRESH_TOKEN`.

Temporary retrieval path until Redis/CLAP lands: keep the Claude-facing `search_track_embeddings` MCP contract unchanged, but internally search the user's Spotify playlists first and Spotify global search second. Returned tracks are normalized into the runtime `Track` shape, registered in the app-owned catalog, marked with `source: "spotify_playlist_search"` and `temporary_until_embeddings: true`, and can immediately flow through `replace_queue` and `play_track` by id.

Live user demo: `uv run python -m claude_dj` runs a bounded end-to-end demo without requiring Claude SDK auth. It loads `.env`, ensures a Spotify Connect device is active by transferring to the first unrestricted device if needed, infers a starting direction from Spotify playlists/current context instead of requiring a genre, searches candidates through the runtime path, generates Deepgram narration, plays that audio locally with macOS `afplay`, starts Spotify playback, then confirms current playback through Spotify. The success marker is `demo: ok`. `--query` exists only as a manual override for experiments, not for the real demo flow.

Long-running harness validation: `uv run python -m claude_dj.main` loads `src/backend/.env` at CLI startup and runs quietly by default with lifecycle-level messages only. Use `uv run python -m claude_dj.main --verbose-claude` or `CLAUDE_DJ_VERBOSE_CLAUDE=1` when debugging; verbose mode prints the full Claude SDK stream for each turn, including system/init messages, assistant text, tool uses, tool results, rate-limit status, and result status. This observability is required for harness debugging; do not replace broken Claude/tool behavior with fallback paths. The startup path has been live-tested through Claude SDK -> DJ MCP -> Spotify playlist/search retrieval -> `replace_queue` -> Deepgram `narrate` with local audio playback -> Spotify `play_track`; follow-up planning now happens only for thresholded reaction events, max-cluster policy events, or empty-queue refresh events.

Claude Code SDK fast mode is opt-in with `CLAUDE_DJ_CLAUDE_FAST_MODE=1`, which passes the CLI `--bare` flag through `ClaudeAgentOptions.extra_args={"bare": None}`. Do not implement fast mode by lowering reasoning `effort`; `effort` remains a separate model behavior knob. `--bare` starts minimal mode and skips hooks, LSP, plugin sync, auto-memory, background prefetches, and keychain/OAuth reads, so leave it off for local OAuth/keychain-authenticated demo runs unless `ANTHROPIC_API_KEY` or an `apiKeyHelper` setting is configured. Claude SDK result errors are now raised instead of silently allowing the harness to keep looping after a failed turn.

Demo pacing: the long-running harness defaults to `--demo-track-seconds 20` and supports `CLAUDE_DJ_DEMO_TRACK_SECONDS=20` to cap each track's effective playback duration. The cap does not alter Redis metadata or ask Claude to skip. `InMemoryPlaybackRuntime.get_current_playback()` reports the capped duration and `seconds_remaining=0` once the cap elapses, so `TrackBoundaryWatcher` advances through the normal deterministic boundary path. For demo timing, elapsed progress is app-owned audible playback time since `play_track`, not Spotify progress alone; explicit `pause_music()` intervals for prepared bridge narration are subtracted so the first post-bridge song gets the same audible demo duration as later songs. The demo queue is capped to 2-4 tracks by `CLAUDE_DJ_QUEUE_MIN_TRACKS` and `CLAUDE_DJ_QUEUE_MAX_TRACKS`; extra replacement candidates are dropped by the runtime.

Spotify device activation belongs in the playback runtime, not only in one-off smoke scripts. Before `play_track`, the runtime checks current playback for an active unrestricted device; if none exists, it lists Spotify Connect devices, transfers playback to the remembered or first unrestricted device, stores that device id in memory, and then starts the track. This keeps Claude's `play_track` tool working without requiring Claude to manage `device_id`.

## Redis data model

Current live Redis Cloud database is `sugar-daylit-corn-40583.db.redis.io:18497`, Redis 8.4, RESP3-capable, plaintext `redis://` on that port, with Search, JSON, Time Series, and probabilistic modules enabled. The developer MCP URL should use `redis://default:${REDIS_PASSWORD}@sugar-daylit-corn-40583.db.redis.io:18497/0`.

Runtime recommendation bridge note: `redis-py` 8 defaults to RESP3 and enables maintenance notifications by default. Against this database, `redis-py` hangs on binary vector fields/params in this environment (`HGET embedding`, `HGETALL` with `embedding`, and `FT.SEARCH ... PARAMS vec`), while the same commands succeed over raw Redis protocol. `claude_dj.mcp.recommendations.RedisRecommendationClient` therefore uses a minimal raw RESP client for the vector recommendation and replay-guard path and authenticates with `HELLO 2 AUTH` to receive RESP2-style flat replies. Keep KNN metadata fetches at `K <= 10`; larger KNN requests with returned metadata timed out against the live demo database. The raw client retries read-only commands up to three times because the public endpoint occasionally times out on initial TCP connect. For recent-track replay reads, avoid `ZRANGEBYSCORE ... WITHSCORES`; the live Redis Cloud endpoint timed out on that command shape. Use raw `ZRANGEBYSCORE ... LIMIT` to fetch members and raw `ZMSCORE` to fetch timestamps.

Track profile:

- track id
- title
- artist
- artwork URL
- Spotify track id
- Spotify URI
- CLAP audio embedding
- embedding provenance
- cluster id if computed
- play/reaction stats

Session state:

- session id
- current track
- current queue
- recent tracks
- current cluster
- cluster streak
- min cluster run: demo default 2
- max cluster run: demo default 4
- current DJ status

Reaction trace:

- timestamp
- track id
- presence score
- movement score
- face/emotion score
- playback signal
- aggregate reaction score

Memory:

- liked clusters
- disliked clusters
- recent skips
- yesterday's genres
- session summaries
- tracks to avoid replaying

Session history:

- session id
- date/time range
- played tracks
- dominant genres/clusters
- reaction highlights
- summary text
- summary embedding
- per-track play events where useful

## Embedding pipeline

Goal: build track vectors from audio using CLAP, with Spotify metadata stored alongside the vector for display, filtering, and ranking.

Prep-time flow:

1. Read Spotify playlists and track metadata.
2. For each track, collect title, artist, album, release year, popularity, playlist context, available artist/genre metadata, and artwork.
3. Obtain audio suitable for CLAP embedding. The implementation path is TBD.
4. Generate a CLAP audio embedding.
5. Normalize the final vector.
6. Store vector, Spotify identifiers, metadata, and embedding provenance in Redis.

Runtime should not depend on embedding generation. The demo should query already-built vectors.

Use CLAP audio embeddings instead of lyric-based text embeddings. This should represent musical similarity more directly than metadata or lyrics alone, while still letting Redis perform fast vector retrieval.

Recommended stack:

- Python worker
- Spotify Web API for playlist, track, artist, album, and playback metadata
- CLAP embedding model for audio vectors
- Redis vector index for storage and retrieval

Spotify and audio caveats:

- Spotify does not provide native embeddings.
- Spotify's Web API does not expose full-track audio files for arbitrary embedding.
- The legal and technical path for obtaining audio inputs is unresolved.
- Treat CLAP embedding generation as a prep-time pipeline, not a runtime dependency.
- Keep a fallback playlist with verified embeddings.
- Store derived vectors and source metadata by default, not retained raw audio.

## Listening-history import

`recommendation_engine.import_history` seeds startup recommendations from the user's last played track without changing the runtime recommendation path. A provider reads one external history item, maps it to `ExternalTrack`, reuses Deezer enrichment, CLAP embedding, Redis track storage, and `recommend.next_five`, then publishes the resolved seed id to Redis key `claudedj:initial_seed_track_id`.

Current provider: `SpotifyHistoryProvider` reads `GET /me/player/recently-played?limit=1`, which requires `user-read-recently-played` on `SPOTIFY_REFRESH_TOKEN`. Use `recommendation_engine/authorize_and_save.py` when re-authorizing because the current working Spotify redirect is `https://127.0.0.1:8888/callback`.

Imported history tracks are ephemeral. The imported `track:{deezer_id}` hash and seed pointer both use a 1 hour TTL, so the shared Redis catalog is not permanently polluted and the harness falls back cleanly when the pointer expires.

The live import path uses a minimal raw RESP2 Redis client for binary vector writes and the recommendation check. This matches the backend recommendation bridge and avoids Redis 8 / redis-py RESP3 timeouts seen on binary `HSET` replies.

Backend startup reads the seed with precedence: `CLAUDE_DJ_INITIAL_REDIS_TRACK_ID`, imported Redis seed, then default `deezer:100814018`. `get_session_context.initial_seed_track_id` exposes that resolved seed to Claude, and the `on_start` prompt tells Claude to prefer it before asking for generic seed candidates.

## Decision flow

Autonomous startup:

1. CLI starts the long-running DJ harness with no required user input.
2. Harness resolves startup seed context from explicit env override, imported last-played history in Redis, current playback if any, and demo defaults.
3. Claude calls `get_session_context`.
4. Claude calls `search_track_embeddings`.
5. Claude selects a coherent 2-4 song demo set.
6. Claude calls `replace_queue` with only that set.
7. Claude calls `narrate` to greet the user and explain the starting direction.
8. Claude calls `play_track`.

Event-driven preparation:

1. While the current song keeps playing, the harness checks deterministic boundary state, then polls the local reaction source and cluster policy monitor.
2. `ReactionMonitor` emits an event only after sustained negative feedback, currently 5 seconds above confidence threshold and outside cooldown.
3. `ClusterPolicyMonitor` picks a target between `CLAUDE_DJ_MIN_CLUSTER_RUN` and `CLAUDE_DJ_MAX_CLUSTER_RUN`, currently 2-4 songs, and emits `max_cluster_streak_reached` when the current group reaches that target. This lets the harness freshen or shift after each short demo group without waiting for negative feedback.
4. If a shift event occurs at or after 75% progress, preparation is deferred to the following song instead of risking a late bridge.
5. For actionable events, Claude calls `get_current_playback`, `get_session_context`, optionally `get_reaction_signal` once, `search_track_embeddings`, `replace_queue(timing="after_current_track")`, and `narrate(mode="prepare", timing="after_current_track")`.
6. `narrate(mode="prepare")` should pre-render/cache narration audio and return an id/readiness result. The current song must not pause while this happens.

Current implementation note: `DJToolHandlers` accepts an optional `ReactionSource`. The long-running `claude_dj.main` harness now builds a `ReactorReactionSource` by default, starts the webcam-backed `WebcamWorker` before Claude's startup turn, and pumps a VibeDJ OpenCV HUD from the main asyncio loop. The HUD was ported from `origin/mood-detection:vibe_dj/main.py` and replaces the legacy text-only `ClaudeDJ Emotion Detection` preview. The visible HUD refreshes from raw camera frames at a separate near-realtime preview cadence, while expensive reaction analysis remains throttled by `sample_interval`. Set `CLAUDE_DJ_NO_WEBCAM=1` to run the reactor without webcam capture; this remains a real reactor source, not the old neutral stub. The webcam path uses DeepFace 7-class emotion distributions plus MediaPipe face landmarks, collapses emotions to `happy` / `neutral` / `disinterested` with track energy and valence context, and weights landmark-derived smile, mouth-open, eye aspect ratio, brow height, face-area, and head-pose patterns more heavily than noisy CNN emotion alone. The path requires the `reactions` extra, including `tf-keras` for DeepFace/TensorFlow compatibility, and `face_landmarker.task` at `CLAUDE_DJ_FACE_LANDMARKER_MODEL` or the default `src/backend/face_landmarker.task` path. The default local model asset is installed there and ignored by Git.

Track-boundary execution:

1. At the boundary, do not call Claude, Redis search, embedding search, or Deepgram.
2. If a ready transition plan matches the ending track, the player starts the prepared next track, pauses music playback, plays the prepared narration audio immediately, then resumes music playback.
3. If no ready plan exists, continue with the next track from the app-owned queue without narration.
4. Stale transition plans must be ignored using track ids or a transition id.

Current implementation note: the long-running CLI harness polls current playback with `TrackBoundaryWatcher`. When `seconds_remaining` reaches zero and a queued or pending next track exists, it calls the deterministic boundary executor exactly once for that track. It does not consume a boundary while both queues are empty; that lets a just-started queue refresh fill `pending_queue_track_ids` before playback advances. It also handles Spotify's natural-end reset state: if the same app-owned track was previously playing, then Spotify reports it stopped with `progress_ms=0` while ClaudeDJ still has queued or pending tracks, the watcher treats that as the missed track boundary. The no-plan fallback calls `InMemoryPlaybackRuntime.play_next_queued_track()` so a normal startup queue can advance without Claude, Redis, or Deepgram on the boundary path. Queue state is logged on each scheduler decision as `current`, `queue`, `pending`, and `seconds_remaining`.

Boundary transitions fade the active Spotify volume down over 1 second before starting the next track, then fade back to the original volume over 1 second. Prepared bridge narration starts the prepared track at zero volume, pauses it for narration, resumes it, then fades back in.

Live agentic E2E validation: `CLAUDE_DJ_LIVE_E2E=1 uv run python -m unittest tests.test_live_agentic_pipeline.LiveAgenticPipelineTests.test_real_claude_session_responds_to_fake_camera_and_plays_three_fast_tracks` runs a real Claude SDK session against the project MCP server, Redis recommendations, Spotify playback, and Deepgram TTS. The only fake input is the camera/reaction source. The test uses a 15-second virtual Spotify duration per song, queues three tracks, flips the fake camera to strong negative, verifies Claude prepares bridge narration, then executes the prepared boundary and deterministic fallback playback for the remaining shifted songs.

End of session:

1. Claude calls `summarize_session`.
2. Redis stores compact memory: genres/clusters played, liked sounds, disliked sounds, fatigue signals.
3. Redis stores a searchable session-history record with timestamped metadata and a summary embedding.

## Context management

Claude should receive compact context, not raw logs.

`get_session_context` should return:

- current track
- seconds remaining
- current queue
- current reaction score
- recent reaction trend
- recent tracks
- recent track replay window and track ids still inside that window
- cluster streak
- liked clusters
- disliked clusters
- yesterday's genres
- recommended next action if available

Redis keeps the full event trail. Claude sees only the decision bundle.

Runtime replay guard: the backend keeps app-owned play timestamps and rejects direct playback of any track played in the last hour. Redis stores the shared replay window in `claudedj:recent_tracks` as a sorted set with track ids scored by wall-clock play timestamp and a one-hour TTL. Each harness hydrates that window once at startup/first use, then `replace_queue` drops one-hour-recent tracks before accepting a queue and `search_track_embeddings(exclude_recent=true)` passes those ids into the existing recommendation exclusion list. This avoids per-recommendation Redis history lookups while preserving the replay guard across backend restarts.

## Session history search

Redis should support two kinds of history lookup:

- Structured lookup: filter by date range, session id, artist, genre, cluster, or reaction score.
- Semantic lookup: embed session summaries so Claude can search natural language questions like "what did I listen to last week?" or "when did I like smoother reggaeton?"

Example records:

- `session:2026-06-20:summary`
- `session:2026-06-20:tracks`
- `user:default:history_index`

Example tool:

- `search_session_history`: query Redis for past sessions by date range, semantic text, filters, or all three.

This gives the DJ memory beyond the current queue without forcing Claude to keep old sessions in its context window.

## Desktop Mascot App

The current frontend surface starts as a mascot-first desktop app:

- ClaudeDJ mascot appears on app startup.
- The mascot is rendered in a transparent, frameless Electron window positioned near the macOS Dock.
- Current implementation uses transparent WebM mascot states and supports horizontal pointer dragging by moving the native app window.
- The Electron main process moves the native window left and right near the Dock, pauses between walks, then chooses another destination. The renderer swaps between idle bob WebMs and a 50/50 random walk-or-skate WebM for the chosen direction; skate WebMs are scaled to 144% size only for that traversal. CSS keyframes do not drive movement.
- The Electron window uses a larger transparent stage sized for the 144% skate animation, while normal, sleep, speaking, and walk assets keep a 190px visual box anchored at the bottom center. This prevents skate clipping without making every mascot state larger.
- Auto-walk and manual drag are clamped to a centered Dock travel lane with an inset, not the full display width, so the full mascot window stays inside the Dock's left/right bounds.
- The native window sits 24px lower than the nominal Dock edge to compensate for transparent baseline padding in the WebM frames.
- Each idle pause randomly chooses between the normal bob and wink bob WebMs.
- Clicking or dragging the mascot plays a short high-pitched, pixelated generated Web Audio "ouch" with randomized pitch, a yelp attack, and a falling tail; no separate audio asset is required.
- Backend-owned mascot startup begins in the transparent sleeping WebM state (`claude-dj-mascot-sleeping-transparent.webm`), with that sleep state translated 16px downward to align it with the Dock. When actual narration audio playback starts, the narration player tells Electron to switch to the singing microphone WebM state (`claude-dj-mascot-singing-transparent.webm`), which also has a 16px downward Dock alignment offset, then returns to normal walking after playback completes. Prepared narration generation alone does not trigger the singing animation.
- `npm run app` and `npm run dev` start the Electron mascot through a launcher that owns the child process lifecycle, enforces one visible mascot instance, and shuts the mascot down when the app command exits.
- The long-running backend harness starts the same Electron launcher as a child process and stops it in the harness shutdown path, so the mascot appears when the app runs and disappears afterward.
- The same backend harness also starts the webcam reaction model and OpenCV emotion preview by default before the Claude startup turn, so judges can see the CV signal path is live.
- Do not use CSS keyframe animation for mascot walking; animated walking should come from a prepared asset.
- Do not render a website-style page, fake desktop, fake Dock, playback controls, chat input, queue editor, or visible technical controls.

Playback metadata can later be layered into the same Dock surface:

- album art
- title
- artist
- small status line
- optional progress bar

Example statuses:

- `listening`
- `staying close · 2/6`
- `shifting after this`
- `reading the room`

Keep this read-only and ambient.

## Reliability constraints

- Keep the queue populated before the current song ends.
- Do not block playback on embedding generation.
- Do not block playback on narration.
- Do not put Claude on the track-boundary critical path.
- Restore music volume after narration even if narration playback fails.
- Use deterministic fallback behavior if Claude is slow.
- Keep a fallback playlist with verified metadata and CLAP embeddings.
- Store derived vectors and source metadata by default, not retained raw audio.

## Parallelizable tasks

Three people can work in parallel after agreeing on the MCP tool contracts and Redis key shapes.

Person 1: agent and MCP harness

- Build the MCP server.
- Implement the core tools: `search_track_embeddings`, `get_session_context`, `replace_queue`, `play_track`, `narrate`, and `mark_track_feedback`.
- Configure Claude Code SDK with the DJ mission prompt.
- Add a deterministic fallback if Claude is slow.

Person 2: camera feedback and playback signals

- Build the webcam reaction worker.
- Produce a simple reaction score from presence, motion, face/emotion, and optional singing/humming cues.
- Watch playback progress, skips, starts, and endings.
- Write reaction events and traces into Redis.

Person 3: Redis, embeddings, and retrieval

- Build the Spotify playlist ingestion loop.
- Build or integrate the TBD CLAP audio embedding pipeline.
- Generate CLAP embeddings for tracks in the recommendation pool.
- Create the Redis vector index and track profile store.
- Implement candidate retrieval, cluster streak state, recent-track exclusion, and ranked candidate output.

Shared integration:

- Agree on the `get_session_context` response shape first.
- Keep the mascot app read-only and minimal.
- Run a dry demo with a verified fallback playlist before adding more tracks.

## Demo story

The judge sees a tiny mascot app near the Dock. The presenter explains the invisible loop:

The user hears music.
The backend watches reactions.
Redis remembers what happened.
Vector search finds possible next songs.
The DJ ranks them using recent context.
Then the player plays the next song.

Then the demo shows a positive reaction causing similar tracks to be queued, and a negative reaction causing the DJ to shift away.
