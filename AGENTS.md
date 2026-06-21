---
project: claude-dj
purpose: onboarding for Claude and GPT-5.5 agents joining the Berkeley AI hackathon project
primary_docs:
  - contexts/spec.md
  - contexts/design.md
  - contexts/claude-dj.md
  - contexts/example-flow.md
  - contexts/hackathon-tracks.md
---

# ClaudeDJ project guide

## First step

Read the full `contexts/` directory before making product, architecture, sponsor, or implementation decisions. The context files are the project knowledge base and should be treated as the source of truth.

Current context files:

- `contexts/spec.md`: product behavior, user experience, success criteria, non-goals.
- `contexts/design.md`: technical architecture, MCP tools, Redis data model, embedding pipeline, session history search, reliability constraints, and parallel work split.
- `contexts/claude-dj.md`: shortest statement of the core loop.
- `contexts/example-flow.md`: simulated user and ClaudeDJ lifecycle.
- `contexts/hackathon-tracks.md`: sponsor and general track knowledge base.

Continuously improve the knowledge base. Whenever you discover new implementation details, sponsor constraints, API limitations, demo decisions, or corrected assumptions, update the relevant file in `contexts/` immediately. Keep the knowledge base compact, current, and useful for the next agent.

## Project summary

ClaudeDJ is a minimal, emotion-aware DJ for the Berkeley AI hackathon. The user sees a small draggable mini player, similar to a Spotify mini player. Behind the scenes, Claude Code SDK drives a DJ agent through an MCP server. Redis stores vectors, state, memory, session history, event streams, and reaction traces.

## Current branch context

We are currently working in the `dj-harness` branch. This branch owns workstream 3: the Claude Code SDK harness and the project MCP layer that integrates playback, recommendations, Redis-backed context, and realtime reaction signals.

Assume two other teammates are working in parallel:

- One teammate owns embeddings, Redis integration, vector indexes, session history storage, and recommendation retrieval internals.
- One teammate owns realtime face detection, reaction scoring, sound analysis, and the worker that emits reaction/playback signals.
- This branch owns the harness and MCP integration surface. Prefer defining contracts, typed tool shapes, stubs, mocks, and orchestration glue here rather than taking over the other two workstreams.

When opening additional agents in this branch, onboard them with this split first. Keep them focused on harness/MCP work unless the user explicitly asks to cross into embeddings, Redis internals, face detection, or sound analysis.

Core loop:

The user hears music.
The backend watches reactions.
Redis remembers what happened.
Vector search finds possible next songs.
The DJ ranks them using recent context.
Then the player plays the next song.

## Current product direction

- Use Spotify for playlists, playback, track metadata, artist metadata, album metadata, and artwork.
- Spotify does not provide native song embeddings through the Web API.
- Use CLAP audio embeddings for tracks in the recommendation pool.
- The exact path for obtaining audio suitable for CLAP embedding is TBD.
- Store derived vectors, source metadata, and embedding provenance in Redis.
- Use Redis vector search for recommendations and session-history search.
- Keep the UI minimal: album art, title, artist, compact status line, optional progress bar.
- Do not add skip controls, queue editing, or a large dashboard unless the user explicitly changes direction.

## Agent harness

The harness is:

- Claude Code SDK as the agent runtime.
- A project MCP server as the tool layer.
- Redis as memory, vector retrieval, state, streams, and history search.
- A draggable mini player as the user-facing surface.

Claude should manage the queue proactively:

- The DJ harness is autonomous. Do not design the runtime around a user request or chat input trigger.
- Startup should use configured seed context, demo defaults, current playback, history, or available signals rather than waiting for user input.
- On startup, choose an initial 1-2 song demo set; do not keep extending the queue immediately.
- Search embeddings before narration.
- Narrate before starting playback and when changing direction.
- Let the harness trigger Claude from thresholded reaction or cluster-policy events; do not poll Claude for neutral mid-song checks.
- If feedback is positive, stay with the current set; future versions can choose a similar next set when needed.
- If feedback is negative and the DJ switches genres/clusters, pre-render bridge narration before the current song ends.
- At the track boundary, do not call Claude. Execute only a ready transition plan or a deterministic fallback.
- During prepared bridge narration, pause music playback, play the narration, then resume playback afterward.
- Do not wait until the song ends to decide what comes next.

## Expected MCP tools

Minimum useful tool set:

- `search_track_embeddings`
- `get_current_playback`
- `get_reaction_signal`
- `replace_queue`
- `play_track`
- `narrate`
- `mark_track_feedback`
- `get_session_context`
- `summarize_session`
- `search_session_history`

Nice-to-have tools:

- `get_cluster_streak`
- `set_cluster_policy`
- `get_yesterday_summary`
- `record_demo_event`

## Redis responsibilities

Use Redis beyond caching:

- Vector search for CLAP audio embeddings.
- Vector or semantic search over session summaries.
- JSON or hashes for track profiles, current session, queue state, and memory.
- Streams for playback, reaction, queue, and narration events.
- Time Series for reaction and engagement traces.
- Sorted sets for candidate ranking.
- Searchable session history for questions like "what did I listen to last week?"

## Recommendation behavior

- Stay in a working music cluster for at least 3 songs.
- Leave a music cluster after the configured max run. The demo harness defaults to 2 songs; the longer product target is 6 songs.
- Strongly negative feedback can break the minimum early.
- Positive feedback should pull the queue toward similar tracks.
- Negative feedback should mark the cluster as disliked and shift away.
- Neutral feedback can trigger a slight shift after the minimum run is satisfied.

## Hackathon strategy

Primary target:

- Redis track. The project uses Redis for vector search, memory, session history, state, event traces, and context retrieval.

Secondary target:

- Claude / Anthropic. Claude Code SDK is the DJ agent runtime and uses MCP tools to manage queue decisions.

General track:

- Ddoski's Playground. The product is creative, playful, demoable, and easy for judges to understand.

Possible bonus:

- Ddoski's Toolbox, if the MCP server and reusable agent harness are polished enough to present as a tool.

Avoid chasing weak fits unless new sponsor information changes the strategy.

## Parallel work split

This project is intended for three people working in parallel.

Person 1: agent and MCP harness

- Build the MCP server.
- Implement the core tools.
- Configure Claude Code SDK with the DJ mission prompt.
- Add deterministic fallback behavior if Claude is slow.
- Current branch: `dj-harness`.

Person 2: camera feedback and playback signals

- Build the webcam reaction worker.
- Produce a simple reaction score from presence, motion, face/emotion, and optional singing/humming cues.
- Watch playback progress, skips, starts, and endings.
- Write reaction events and traces into Redis.
- Treat this as an external integration dependency from the `dj-harness` branch unless asked otherwise.

Person 3: Redis, embeddings, and retrieval

- Build the Spotify playlist ingestion loop.
- Build or integrate the TBD CLAP audio embedding pipeline.
- Generate CLAP embeddings for tracks in the recommendation pool.
- Create the Redis vector index and track profile store.
- Implement candidate retrieval, cluster streak state, recent-track exclusion, ranked candidate output, and session history search.
- Treat this as an external integration dependency from the `dj-harness` branch unless asked otherwise.

Shared integration:

- Agree on the `get_session_context` response shape first.
- Keep the mini player read-only and minimal.
- Run a dry demo with a verified fallback playlist before adding more tracks.

## Sentry MCP development workflow

- Use the Sentry MCP during development debugging and verification. OpenCode is configured in `opencode.json` with the remote Sentry MCP at `https://mcp.sentry.dev/mcp`; Claude-style clients are configured in `.mcp.json`.
- If Sentry tools are unavailable, run `opencode mcp list` and authenticate with `opencode mcp auth sentry`. Restart OpenCode after config changes because MCP config is loaded at startup.
- Current Sentry org/project for this repo: org `pennsylvania-state-universi-og`, project `javascript`.
- MCP-first rule: when investigating crashes, failed builds, source-map issues, or demo instability, check recent Sentry issues/traces through MCP before guessing from logs alone.
- Demo-critical issue alerts are configured in Sentry as `ClaudeDJ backend demo-critical errors` and `ClaudeDJ frontend demo-critical errors`. They target events tagged `service=claude_dj_backend` and `service=claude_dj_frontend`.
- The backend emits spans/breadcrumbs around Claude SDK turns, MCP tool calls, Deepgram narration generation, and track-boundary transitions. Preserve these tags when changing those paths.
- Do not print or commit `SENTRY_AUTH_TOKEN`. It belongs in local env or CI for source-map uploads; DSNs are public and are documented in the env examples.

## Documentation rules

- Keep context docs concise and current.
- Prefer updating existing `contexts/` files over adding scattered new notes.
- Record corrected assumptions immediately.
- Do not leave stale references to old architecture choices.
- When APIs or sponsor requirements are uncertain or current, verify with primary sources before updating the knowledge base.

## Lessons

- Do not frame ClaudeDJ as request-driven. The harness should be an autonomous long-running DJ loop that starts from configured context and signals, not a `user_request` event or chat prompt.
- Keep backend harness code modular: `claude_dj/main.py` for the script entrypoint, `agent/` for Claude SDK lifecycle code, `mcp/` for project MCP tools and narration, `transition.py` for boundary execution, and markdown prompts under `agent/prompts/` with YAML frontmatter plus simple XML sections.
- For Deepgram narration, keep `immediate` mode in the tool contract for startup narration, but avoid adding asset caches or error-hardening by default. Scope the first audio implementation to the smallest path the demo needs.
- Deepgram Aura-2 REST supports `speed` but not a general emotion/style knob. For a more excited DJ sound, tune model, speed, and narration copy before adding extra machinery.
- When a user asks to install an MCP server "here," clarify whether they mean project-local MCP config or user-level local MCP config before editing project files.
- For Spotify OAuth, do not recommend `localhost` redirect URIs. Current Spotify docs require explicit loopback IP redirect URIs such as `http://127.0.0.1:8888/callback`, and the dashboard, authorize URL, and token exchange redirect URI must match exactly.
- Do not make the user-facing ClaudeDJ demo CLI start from a fixed genre or query. The real demo should feel autonomous: infer a starting direction from available Spotify playlists/current context/defaults, narrate the choice, and play music without requiring a user seed.
- Do not add fallback paths when debugging the ClaudeDJ harness unless explicitly requested. The original Claude SDK -> MCP -> Spotify/Deepgram loop must work unbroken; add observability and fix the actual failing step instead of masking it.
- When reframing generated mascot assets, use deterministic crop/scale operations instead of AI redraws. Redraw-based reframing can silently change fixed geometry such as leg height and body proportions.
- For mascot video generation, do not ask the model to turn or reorient the character unless changing identity is acceptable. Preserve angle/orientation explicitly, and describe only the small local motion needed, such as a front-facing pixel walk in place.
- For sleeping mascot video generation, explicitly require flat 2D pixelated animation, visible Zzz symbols, a stationary creature, unchanged background, unchanged orientation, and no camera expansion/shrinkage; include style words like "bubbly" only when the user requests them.
- When clearing mascot video backgrounds, verify transparency by extracting the alpha mask. AI background removal can leave off-white interior holes opaque; clean those with deterministic RGB colorkeying before shipping.
- When replacing mascot state assets, update the actual Electron renderer state reference too. A transparent WebM in `public/` will not affect the app if `renderer.html` still points that state at an older PNG with an opaque background.
- Do not implement the ClaudeDJ mascot interface as a website. It should be a desktop app surface: a transparent, frameless app window that renders the mascot near the macOS Dock and moves the native window for dragging/walking.
- When updating Redis MCP config for ClaudeDJ, point `.mcp.json` at the Redis Cloud URL with `${REDIS_PASSWORD}` unless the user explicitly asks for a local Redis instance. Do not leave the Redis MCP server on `localhost` for the shared demo database.
- When the user asks to bound the mascot to the Dock, treat it as a horizontal left/right travel constraint unless they explicitly mention vertical placement. Do not change the baseline compensation or vertical Dock offset while tuning horizontal bounds.
- Do not equate Claude Code SDK fast mode with lower reasoning effort. Inspect the actual installed SDK/CLI option surface before changing model, effort, or extra args for performance-related requests.
- Keep the long-running harness event-driven: `ReactionMonitor` owns sustained negative timing, `ClusterPolicyMonitor` owns max-cluster auto-rotation, late shift events at or after 75% progress defer to the following song, and every genre/cluster shift must prepare bridge narration before the boundary.
- Keep queue observability visible in the live harness. Log `current`, active queue, pending queue, and seconds remaining on scheduler decisions; do not consume a track boundary while both active and pending queues are empty, because queue refresh may still be preparing `pending_queue_track_ids`.
- Count demo track duration with app-owned audible playback time from `play_track`, not Spotify progress alone. Prepared bridge narration pauses Spotify, so subtract explicit `pause_music()` intervals; otherwise the first post-bridge track runs narration duration plus the demo cap or gets skipped if raw wall-clock is used.
- When a user reports a possible live behavior issue and then uncertainty appears, verify the behavior before changing timing semantics. Do not turn an uncertain signal into a deterministic scheduler change without a clear reproduction.
- Never delete untracked generated media assets during cleanup unless the user explicitly names those files for deletion. Untracked assets cannot be restored from Git, and public mascot animation candidates may be intentionally kept even when not referenced by code.
