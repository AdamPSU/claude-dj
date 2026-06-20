
# ClaudeDJ 24-Hour Hackathon Timeline

## Context

3-person team, 24-hour hackathon (June 20-21). Goal: demoable CLI MVP by hour 10-12 that shows the full core loop working. Then spend remaining hours adding webcam, UI polish, and judge-ready demo.

**CLI MVP = the core loop working in a terminal:**
Music plays via Spotify -> user gives CLI feedback (like/dislike/meh) -> Redis remembers -> vector search finds next tracks -> Claude agent decides -> queue updates -> DJ narrates in terminal.

---

## Phase 0: Setup & Contracts (Hour 0-1) — ALL THREE

**Everyone together for the first hour. Don't split until contracts are locked.**

- [ ] Spotify Developer app created, OAuth tokens working
- [ ] Redis Cloud instance running (or local Redis Stack with vector search)
- [ ] Repo structure agreed on, env vars shared
- [ ] **Lock the `get_session_context` response shape** (this is the integration contract)
- [ ] Lock Redis key naming conventions (e.g., `track:{id}`, `session:current`, `queue:current`)
- [ ] Lock MCP tool input/output schemas for the 6 MVP tools
- [ ] Pick embedding model (e.g., `text-embedding-3-small` from OpenAI, or Voyage)
- [ ] Pick lyrics source (LRCLIB for speed)
- [ ] Seed playlist chosen (50-100 tracks, genre you can demo well)

**MVP tool set (6 tools, not 10):**
1. `search_track_embeddings` — vector search in Redis
2. `play_track` — start Spotify playback
3. `replace_queue` — update Spotify queue
4. `get_session_context` — compact decision bundle from Redis
5. `narrate` — print DJ line to terminal
6. `mark_track_feedback` — write like/dislike to Redis

**Cut from MVP:** `get_current_playback` (fold into session context), `get_reaction_signal` (fold into session context), `summarize_session`, `search_session_history`

---

## Phase 1: Foundation (Hour 1-5) — PARALLEL

### Person 1: MCP Server + Agent Harness
- [ ] MCP server scaffold (Python or TypeScript)
- [ ] Stub all 6 MVP tools with correct schemas
- [ ] Write the DJ mission system prompt
- [ ] Claude Code SDK integration — agent can call tools in a loop
- [ ] Hardcode a few track IDs for dry-run testing (don't wait for Person 3)

### Person 2: Spotify Playback + CLI Interface
- [ ] Spotify Web API auth flow (device playback requires Premium)
- [ ] `play_track(spotify_uri)` working — music actually plays
- [ ] `replace_queue(track_uris)` working — can set upcoming tracks
- [ ] CLI display: show current track name, artist, queue preview
- [ ] CLI input: user can type `like`, `dislike`, `meh` → writes reaction to Redis
- [ ] Playback state polling: detect track changes, write events to Redis

### Person 3: Embeddings + Redis Data Layer
- [ ] Spotify playlist fetch → get track metadata for seed playlist
- [ ] LRCLIB lyrics fetch → get lyrics for each track, flag missing
- [ ] Build text documents per track (title + artist + album + genre + lyrics summary)
- [ ] Generate embeddings via chosen model
- [ ] Create Redis vector index (`idx:tracks`)
- [ ] Store track profiles as Redis JSON with embedded vectors
- [ ] Verify: vector search query returns sensible results
- [ ] Implement `search_track_embeddings` query logic (text query and seed-track modes)

**Hour 5 checkpoint:** Person 1 has an agent that calls tool stubs. Person 2 has Spotify playing music from CLI. Person 3 has 50+ tracks with vectors in Redis and search working.

---

## Phase 2: Wire It Together (Hour 5-8) — INTEGRATION BEGINS

### Person 1: Real Tool Implementations
- [ ] Connect `search_track_embeddings` to Person 3's Redis query code
- [ ] Connect `play_track` and `replace_queue` to Person 2's Spotify code
- [ ] Implement `get_session_context` — read current state from Redis, return decision bundle
- [ ] Implement `mark_track_feedback` — write feedback to Redis hash
- [ ] Implement `narrate` — print to terminal with formatting
- [ ] Test: agent can do one full cycle (search → pick → play → narrate)

### Person 2: Reaction Pipeline
- [ ] CLI feedback writes to Redis stream (`stream:reactions`)
- [ ] Compute simple reaction score from recent feedback entries
- [ ] Playback progress tracking: seconds remaining, track changes
- [ ] Write playback events to Redis (`stream:playback`)
- [ ] Make sure `get_session_context` can pull latest reaction + playback state

### Person 3: Retrieval Logic
- [ ] "Similar" mode: given a seed track, find nearest neighbors excluding recent plays
- [ ] "Shift" mode: find tracks in adjacent clusters, avoid disliked clusters
- [ ] Recent-track exclusion list from Redis
- [ ] Liked/disliked cluster tracking in Redis
- [ ] Basic cluster assignment (can be simple — e.g., genre tag or nearest centroid)
- [ ] Ranked candidate output matching the MCP tool schema

**Hour 8 checkpoint:** Full loop works end-to-end at least once. Agent searches, picks, plays, narrates. User can give feedback. Agent adjusts.

---

## Phase 3: CLI MVP Polish (Hour 8-12) — ALL THREE

**This is the demo-ready push. Focus on reliability, not new features.**

- [ ] Test the full loop 5+ times with real music
- [ ] Positive feedback → queue gets similar tracks (verify)
- [ ] Negative feedback → queue shifts away (verify)
- [ ] Neutral → queue stays stable (verify)
- [ ] DJ narration sounds natural (tune the system prompt)
- [ ] Deterministic fallback: if Claude is slow, play next from pre-ranked candidates
- [ ] Fallback playlist: 20 verified tracks that always work
- [ ] CLI output is clean and readable (current track, queue, DJ status, reaction)
- [ ] Prepare a 2-minute demo script with specific commands to type
- [ ] Test demo script end-to-end twice

**Hour 12: CLI MVP is demo-ready.** You can show a judge:
1. "Play reggaeton" → music starts, DJ narrates
2. Type "like" → queue refreshes with similar tracks
3. Type "dislike" → DJ shifts, narrates the change
4. Redis data visible (show vectors, state, streams)

---

## Phase 4: Enhancements (Hour 12-18) — PARALLEL AGAIN

### Person 1: Agent Intelligence
- [ ] Cluster streak logic (3-6 song rule)
- [ ] Mid-song context checks (agent polls without waiting for user input)
- [ ] Richer narration (reference track names, explain shifts)
- [ ] `summarize_session` tool
- [ ] `search_session_history` tool

### Person 2: Webcam Reaction Worker
- [ ] Webcam capture (OpenCV or MediaPipe)
- [ ] Presence detection (is someone there?)
- [ ] Basic emotion/engagement scoring (smile, head nod, movement)
- [ ] Write webcam signals to Redis alongside CLI feedback
- [ ] Blend webcam + CLI signals into single reaction score

### Person 3: More Data + Mini Player UI
- [ ] Expand track pool (200+ tracks across multiple genres)
- [ ] Session history records in Redis
- [ ] If time: basic web mini player (album art, title, artist, status line)
- [ ] If time: draggable component

**Hour 18 checkpoint:** Webcam working or close. Larger track pool. Agent is smarter about cluster management.

---

## Phase 5: Demo Polish (Hour 18-22)

- [ ] Pick best demo path: CLI-only? CLI + webcam? CLI + mini player?
- [ ] Rehearse the 3-minute pitch
- [ ] One-liner ready: "A Claude-driven DJ agent that uses Redis memory and vector search to adapt music from live reactions"
- [ ] Prepare Redis visualization (show vectors, streams, state in RedisInsight or CLI)
- [ ] Edge case fixes
- [ ] Record a backup demo video in case of Spotify/network issues

---

## Phase 6: Submit (Hour 22-24)

- [ ] Final demo rehearsal
- [ ] Write submission description hitting Redis track criteria + Claude track criteria
- [ ] Submit
- [ ] Sleep

---

## What to Cut If Behind Schedule

| If behind at... | Cut this | Keep this |
|---|---|---|
| Hour 5 | Lyrics in embeddings (use metadata-only vectors) | Vector search working with any embeddings |
| Hour 8 | Cluster logic, shift modes | Basic similar-track search + play |
| Hour 10 | Polished narration | Music plays, feedback changes queue |
| Hour 12 | Webcam entirely | CLI feedback as the demo input |
| Hour 18 | Mini player UI | CLI demo with Redis visualization |

## What NOT to Build for MVP

- No web UI (CLI is the MVP surface)
- No chat interface
- No skip button or manual queue editing
- No raw lyrics storage
- No session history search (just current session)
- No time series traces
- No `get_cluster_streak` / `set_cluster_policy` tools
- No multi-user support
