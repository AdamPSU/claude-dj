# DJ agent technical design

## Architecture

The system has four layers:

- Claude Code SDK: runs the DJ agent with a custom mission prompt.
- MCP server: exposes tools Claude can call.
- Redis: stores vectors, state, memory, event streams, and reaction traces.
- Mini player: shows the current song and compact DJ status.

Claude is the high-level queue manager. Redis is the memory and retrieval layer. The MCP server is the boundary between the agent and the music system.

## Agent harness

Claude Code SDK runs with a DJ-specific system prompt:

- Keep 3-6 tracks queued.
- Search embeddings before narration.
- Narrate before starting playback and when changing direction.
- Check mid-song reaction signals.
- If positive, refresh the queue with similar tracks.
- If negative, mark feedback and replace the queue with shifted tracks.
- Do not wait until the song ends to decide what comes next.
- Keep user-facing narration short.

The agent should act through MCP tools only. It should not directly modify Redis or playback internals.

## MCP tools

Minimum useful tool set:

- `search_track_embeddings`: search Redis vectors by text query, seed track, similarity, or shift mode.
- `get_current_playback`: return current track, progress, seconds remaining, and queue.
- `get_reaction_signal`: return current and recent engagement scores.
- `replace_queue`: replace upcoming tracks with selected candidates.
- `play_track`: start playback from a selected track.
- `narrate`: speak or display a short DJ line.
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

The repo includes a Claude Code project MCP config in `.mcp.json`:

- `sentry`: remote Sentry MCP over HTTP at `https://mcp.sentry.dev/mcp`; authenticate through Claude Code's MCP OAuth approval flow.
- `redis`: official Redis MCP via `uvx --from redis-mcp-server@latest redis-mcp-server --url redis://localhost:6379/0`; change the URL when using Redis Cloud or a non-local Redis instance.
- `deepgram`: Deepgram CLI MCP via `uvx --from deepctl dg mcp --non-interactive`; authenticate with `dg login` or provide `DEEPGRAM_API_KEY` in the environment.

These are developer/client MCPs. The product's custom DJ MCP server remains the runtime tool boundary for playback, retrieval, memory, and narration.

## Redis data model

Track profile:

- track id
- title
- artist
- artwork URL
- Spotify track id
- Spotify URI
- CLAP audio embedding
- cluster id if computed
- play/reaction stats

Session state:

- session id
- current track
- current queue
- recent tracks
- current cluster
- cluster streak
- min cluster run: 3
- max cluster run: 6
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

Goal: build comprehensive track vectors using CLAP audio embeddings.

Prep-time flow:

1. Read Spotify playlists and track metadata.
2. For each track, collect title, artist, album, release year, popularity, playlist context, and available artist/genre metadata.
3. Obtain an audio source for the track (exact source TBD).
4. Generate a CLAP audio embedding from the audio source.
5. Normalize the final vector.
6. Store vector, Spotify identifiers, and metadata in Redis.

Runtime should not depend on embedding generation. The demo should query already-built vectors.

Use CLAP audio embeddings. CLAP (Contrastive Language-Audio Pretraining) produces vectors that capture the acoustic character of a track directly, allowing retrieval by both audio similarity and text queries against the same embedding space.

Recommended stack:

- Python worker
- Spotify Web API for playlist, track, artist, album, and playback metadata
- CLAP model for audio embedding generation
- Redis vector index for storage and retrieval

Spotify caveats:

- Spotify does not provide native embeddings.
- Spotify policy warns against ingesting Spotify content into ML/AI models, so this should be treated as a hackathon demo assumption, not a production-safe plan.

## Decision flow

Initial request:

1. User asks for a vibe.
2. Claude calls `search_track_embeddings`.
3. Claude selects 3-6 tracks.
4. Claude calls `narrate`.
5. Claude calls `play_track` and/or `replace_queue`.

Mid-song update:

1. Reaction worker writes scores to Redis.
2. Claude calls `get_session_context`.
3. Claude decides positive, neutral, or negative.
4. Positive: search similar tracks and refresh queue.
5. Negative: mark current cluster disliked, search shifted tracks, replace queue, narrate.
6. Neutral: keep queue if cluster run is under 3; consider slight shift after 3.

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
- cluster streak
- liked clusters
- disliked clusters
- yesterday's genres
- recommended next action if available

Redis keeps the full event trail. Claude sees only the decision bundle.

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

## Mini player

The UI is a draggable mini player:

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

No skip button, queue editor, or visible technical controls.

## Reliability constraints

- Keep the queue populated before the current song ends.
- Do not block playback on embedding generation.
- Do not block playback on narration.
- Use deterministic fallback behavior if Claude is slow.
- Keep a fallback playlist with verified metadata and CLAP embeddings.
- Store derived vectors and source metadata by default.

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
- Obtain audio sources and generate CLAP audio embeddings per track.
- Create the Redis vector index and track profile store.
- Implement candidate retrieval, cluster streak state, recent-track exclusion, and ranked candidate output.

Shared integration:

- Agree on the `get_session_context` response shape first.
- Keep the mini player read-only and minimal.
- Run a dry demo with a verified fallback playlist before adding more tracks.

## Demo story

The judge sees a tiny music player. The presenter explains the invisible loop:

The user hears music.
The backend watches reactions.
Redis remembers what happened.
Vector search finds possible next songs.
The DJ ranks them using recent context.
Then the player plays the next song.

Then the demo shows a positive reaction causing similar tracks to be queued, and a negative reaction causing the DJ to shift away.
