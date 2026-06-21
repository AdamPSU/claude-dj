# ClaudeDJ Technical Track Pitch

ClaudeDJ is an autonomous, emotion-aware DJ harness. It is not a chatbot that waits for a request. It starts from session context, plays music, watches live reaction signals, searches memory and audio embeddings, prepares the next direction before the song ends, and executes the transition without putting Claude on the realtime boundary path.

The one-line demo story is:

The user hears music. The backend watches reactions. Redis remembers what happened. Vector search finds possible next songs. Claude ranks them using compact context. Deepgram gives the DJ a voice. Sentry shows whether the full system is healthy.

## High-Level Harness

The harness has four main layers:

- Claude Code SDK runs the DJ agent with a custom mission prompt.
- A project MCP server exposes the only tools Claude is allowed to use.
- Redis stores retrieval data, live state, memory, replay guards, and session history.
- The desktop mascot app gives the user a small ambient surface instead of a dashboard or chat UI.

At startup, `uv run python -m claude_dj.main` launches the long-running process. The agent wakes once for `on_start`, reads compact session context, searches candidate tracks, chooses a coherent 2-4 track set, narrates the opening, and starts Spotify playback through MCP tools.

After startup, the harness is event-driven. Reaction and cluster-policy monitors decide when a Claude turn is worth running. Claude is not polled for every neutral mid-song moment. When a shift is needed, Claude prepares the next queue and bridge narration while the current song keeps playing.

The most important reliability decision is the track boundary design. At `on_track_boundary`, the system does not call Claude, Redis search, embedding search, Spotify search, or Deepgram. The boundary executor either uses a ready transition plan or continues deterministically with the app-owned queue. This keeps music playback responsive even if an LLM turn, network call, or TTS request is slow.

The project MCP server currently exposes the DJ tool surface through the in-process `dj` server:

- `get_session_context`
- `search_track_embeddings`
- `get_seed_candidates`
- `replace_queue`
- `narrate`
- `play_track`
- `get_current_playback`
- `get_reaction_signal`
- `mark_track_feedback`
- `summarize_session`
- `search_session_history`

That tool layer is the contract between Claude and the rest of the system. Claude makes DJ decisions; the tool layer owns playback, retrieval, narration, reaction access, memory writes, and observability.

## Redis Alignment

Redis is the memory and realtime context layer, not a cache. The Redis track asks for AI apps that use Redis for agent memory, vector search, fast context retrieval, and stateful behavior. ClaudeDJ is built around exactly that loop.

Redis capabilities used or targeted by the harness:

- Redis Cloud / Redis 8.4 as the shared live database.
- Redis Search / RediSearch `FT.SEARCH` for vector KNN retrieval over CLAP audio embeddings.
- Redis hashes for track profiles, including metadata, Spotify identifiers, cluster tags, ranks, artwork, and binary embeddings.
- Redis sorted sets for the one-hour replay guard in `claudedj:recent_tracks`, so recently played tracks are excluded across backend restarts.
- Redis keys with TTL for ephemeral imported-history seed tracks and seed pointers.
- Redis JSON or hashes for session, queue, and memory state.
- Redis Streams for playback, reaction, queue, and narration events.
- Redis Time Series for reaction and engagement traces.
- Redis-backed session history search so Claude can use previous listening summaries instead of carrying old sessions in its context window.
- The official Redis MCP server in developer/client config, pointed at the shared Redis Cloud URL, for inspection and debugging during development.

The live recommendation bridge uses `RedisRecommendationClient` and a minimal raw Redis protocol client. It issues vector searches such as:

```text
FT.SEARCH idx:tracks "(@genre_tag:{...})=>[KNN K @embedding $vec AS score]" PARAMS 2 vec <binary-vector> SORTBY score ASC DIALECT 2
```

This lets the DJ search by musical similarity instead of only metadata. The corpus strategy uses CLAP audio embeddings from preview audio, stores the derived vectors and provenance in Redis, and uses Spotify metadata/playback for the listener-facing experience.

Redis is also how the agent stays practical. Claude gets a compact decision bundle from `get_session_context`: current track, current queue, reaction trend, cluster streak, recent tracks, liked/disliked clusters, replay-window exclusions, and recommended next action. Redis keeps the full event trail, while Claude sees only what it needs for the next DJ decision.

## Sentry Alignment

Sentry is the observability layer for the whole agentic pipeline. We did not just add error capture. We built Sentry around the core questions a judge or engineer would ask: Did Claude run? Which MCP tools did it call? How slow were recommendations and narration? Did the frontend or backend fail? Which pipeline agent or scenario produced a trace?

Implemented observability:

- Python backend initializes `sentry-sdk` before the autonomous harness runs.
- Next.js frontend uses `@sentry/nextjs` browser, server, and edge configuration.
- Backend events are tagged `service=claude_dj_backend`.
- Frontend events are tagged `service=claude_dj_frontend`.
- Claude lifecycle turns are wrapped as `claude_dj.run` transactions.
- MCP tool calls are wrapped as `mcp.tool` spans with tool name, sequence, session id, run id, and run type.
- Deepgram narration generation is wrapped as `http.client.deepgram` spans.
- Warnings and swallowed exceptions are captured with operation tags and sanitized context.
- Breadcrumbs mark run start/completion and tool start/completion.
- Runtime attribution fields can tag traces with collaboration id, agent id, agent name, workstream, scenario, task kind, and verification id.

We created a Sentry dashboard named `ClaudeDJ Observability`:

- Dashboard ID: `7339119`
- URL: `https://pennsylvania-state-universi-og.sentry.io/dashboard/7339119/?project=-1&statsPeriod=24h`
- Widgets: recent ClaudeDJ runs, MCP tool calls over time, MCP tool calls by tool, backend errors over time, frontend errors over time, backend error groups, and frontend error groups.

We also built a report path and produced a concrete report artifact:

- JSON source: `reports/sentry-agent-report.json`
- PDF output: `reports/sentry-agent-report.pdf`
- Report writer: `src/backend/claude_dj/reporting/sentry_report.py`
- Command shape: `python -m claude_dj.reporting.sentry_report --input report.json --output report.pdf`

The report is titled `A Multi-Agent Sentry Evaluation of ClaudeDJ's Autonomous DJ Pipeline`. It summarizes fifteen named pipeline agents, verification commands, Sentry dashboard evidence, recent backend traces, MCP tool latency, backend error groups, and remaining demo risks. This gives us a technical story beyond "we added monitoring": we used Sentry as an evaluation layer for the agent harness.

Demo-critical alerts are also configured for backend and frontend services. That matters because this system has multiple realtime surfaces: Claude turns, MCP tools, Redis search, Spotify playback, Deepgram TTS, webcam reactions, and the mascot UI. Sentry gives us one place to see which part failed.

## Deepgram Alignment

Deepgram is the voice layer. Claude decides what the DJ should say, but Deepgram turns that decision into a short spoken transition.

The `narrate` MCP tool supports two important modes:

- `immediate`: used for startup narration before playback begins.
- `prepare`: used for mid-song bridge narration when the DJ is changing direction after a reaction or cluster-policy event.

The backend uses Deepgram Aura Text-to-Speech through `/v1/speak`. The current default model is `aura-2-luna-en`, with speed configurable through `DEEPGRAM_TTS_SPEED`. Generated audio is short-lived in memory by narration id rather than persisted as a long-term cache.

This is a strong Deepgram fit because speech changes the product experience. The DJ does not just update a queue silently. It explains why it is starting in a pocket, why it is staying close, or why it is moving one step away. The voice is intentionally short, restrained, and music-guide-like instead of a generic hype announcer.

The prepared narration path is also technically meaningful. Deepgram generation happens before the track boundary. At the boundary, the executor can start the prepared next track, pause music during the bridge line, play the Deepgram audio, and resume playback without waiting on a new TTS request.

## Why This Is The Most Technical Track-Level Build

ClaudeDJ looks playful, but the implementation is a dense systems project. It combines an autonomous agent runtime, MCP tool orchestration, Redis vector retrieval, external playback control, voice generation, computer-vision reaction signals, deterministic realtime transitions, observability, and a desktop mascot surface.

The technical depth is in the coordination:

- The agent is autonomous, not request-driven.
- Claude is allowed to decide music direction, but not allowed onto the realtime boundary path.
- Redis stores vectors, session context, recent playback guards, memory, event traces, and history search data.
- CLAP embeddings make the recommendations audio-aware instead of only text or metadata-aware.
- Spotify handles playback and metadata while ClaudeDJ owns the queue because Spotify does not expose a reliable clear-and-replace queue primitive.
- Deepgram adds low-latency spoken narration at exactly the moments where voice improves the UX.
- Sentry traces the agentic runtime, MCP tool calls, Deepgram spans, frontend errors, backend errors, and multi-agent verification metadata.
- The reaction pipeline turns webcam/emotion/landmark signals into thresholded events so Claude is only invoked when the signal matters.
- The mascot surface is a native-feeling transparent desktop app, not a website dashboard.

Most hackathon projects can demo a single API call or a single model interaction. ClaudeDJ has to keep an actual realtime loop alive. If one part blocks, the song boundary feels broken. If recommendation retrieval is slow, bridge preparation is late. If narration is generated at the wrong time, the transition feels awkward. If observability is weak, the team cannot tell whether Claude, Redis, Deepgram, Spotify, the camera, or the frontend caused the issue.

That is why the project belongs in the most technical conversation at the hackathon. The user-facing object is small: a mascot DJ that plays music and reacts to the room. The system behind it is a multi-service, agentic, event-driven runtime that uses sponsor technology as core infrastructure rather than decoration.
