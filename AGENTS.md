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

- Keep 3-6 tracks queued.
- Search embeddings before narration.
- Narrate before starting playback and when changing direction.
- Check mid-song reaction signals.
- If feedback is positive, refresh the queue with similar tracks.
- If feedback is negative, mark feedback and replace the queue with shifted tracks.
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
- Leave a music cluster after 6 songs to avoid staleness.
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

This project is intended for three people.

Person 1: agent and MCP harness

- Build the MCP server.
- Implement the core tools.
- Configure Claude Code SDK with the DJ mission prompt.
- Add deterministic fallback behavior if Claude is slow.

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
- Implement candidate retrieval, cluster streak state, recent-track exclusion, ranked candidate output, and session history search.

Shared integration:

- Agree on the `get_session_context` response shape first.
- Keep the mini player read-only and minimal.
- Run a dry demo with a verified fallback playlist before adding more tracks.

## Documentation rules

- Keep context docs concise and current.
- Prefer updating existing `contexts/` files over adding scattered new notes.
- Record corrected assumptions immediately.
- Do not leave stale references to old architecture choices.
- When APIs or sponsor requirements are uncertain or current, verify with primary sources before updating the knowledge base.
