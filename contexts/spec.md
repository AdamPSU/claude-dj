# DJ agent product spec

## Summary

Build a minimal, emotion-aware DJ that keeps music playing, watches user reactions, and adjusts the queue using CLAP audio embeddings plus Spotify metadata.

The user sees a tiny draggable desktop mascot near the macOS Dock. The system behind it is an agent harness: Claude Code SDK drives decisions, our MCP server exposes music tools, and Redis stores memory, vectors, state, and retrieval context.

## Core loop

The user hears music.
The backend watches reactions.
Redis remembers what happened.
Vector search finds possible next songs.
The DJ ranks them using recent context.
Then the player plays the next song.

## User experience

- The DJ starts autonomously from configured session context, demo defaults, current playback, history, or available signals.
- A transparent, frameless desktop app window shows the ClaudeDJ mascot near the macOS Dock on app startup.
- The backend webcam reaction model and VibeDJ OpenCV HUD start by default with the long-running harness; `CLAUDE_DJ_NO_WEBCAM=1` is the explicit opt-out. The current reaction model combines DeepFace emotions with MediaPipe landmark-expression features rather than a request-driven or manual feedback-only signal.
- The initial frontend prototype only shows the still mascot; playback metadata can be layered into this app surface later.
- No skip button, queue editor, chat box, or large dashboard.
- The DJ may narrate short transitions, especially when it starts or changes direction.
- Use Deepgram for spoken DJ narration when audio narration is enabled.
- Preferred narration direction: a personal music-guide persona modeled after Spotify DJ's public product language: short contextual commentary, human pacing, restrained confidence, nostalgia plus discovery, and a sense that the DJ understands the listener's moment. Exact Deepgram voice/model remains TBD after auditioning and API verification.
- The system should feel ambient, not like a chat app.

## Primary behavior

- Claude starts from a configured seed vibe, current playback context, or session history.
- Claude searches track embeddings before choosing tracks.
- Claude chooses an initial 2-4 track demo set.
- Claude does not queue more songs beyond that initial set at startup.
- Claude narrates the starting choice.
- Playback begins.
- Mid-song reaction signals are collected by the harness and thresholded before Claude is called.
- If the user seems to like the genre/cluster, Claude can keep the current set going.
- If the user seems not to like the genre/cluster, Claude prepares a shifted set and pre-renders bridge narration in the background while the current song continues.
- Tracks played by current or recent harness sessions in the last hour must not be replayed.
- At the track boundary, playback starts the prepared next direction immediately, pauses music during prepared narration, then resumes playback.
- If the signal is neutral, the DJ can make a slight shift after the minimum run is satisfied.

## Similarity run rule

- Stay in a working music cluster for the current randomly selected 2-4 song group unless feedback is strongly negative.
- Leave or freshen a music cluster after the selected group target is reached; the demo harness picks a new target between 2 and 4 songs per group.
- A strongly negative reaction can break the minimum early.
- The current cluster streak is part of session context.

## Embedding strategy

- Use Spotify for playlists, playback, track metadata, artist metadata, album metadata, and artwork.
- Spotify does not provide native song embeddings through the Web API.
- Use CLAP to create audio embeddings for tracks in the recommendation pool.
- The exact path for obtaining audio suitable for CLAP embedding is TBD.
- Store the derived audio vector, source metadata, and embedding provenance in Redis.
- Use Redis vector search for audio-similarity recommendations.
- Store session history in Redis so the DJ can answer and use questions like "what did I listen to last week?"

## Redis usage

- Vector Search: retrieve songs by CLAP audio embedding similarity, and search session history.
- JSON or hashes: store track profiles, current session, and queue state.
- Streams: route playback, reaction, queue, and narration events.
- Time Series: store reaction and engagement traces over time.
- Sorted sets: rank candidate tracks.
- Memory/context records: store recent songs, one-hour replay guard state, liked clusters, disliked clusters, yesterday's genres, and cluster streaks.
- Session history: store searchable listening summaries, played tracks, reactions, and time ranges.

## Claude / agent usage

- Claude Code SDK is the agent runtime.
- A custom DJ mission prompt tells Claude how to manage the queue.
- Our MCP server gives Claude tools for playback, retrieval, memory, and narration.
- The `narrate` tool should produce a short display line and, when available, Deepgram TTS audio.
- Claude should not wait for a song to end before acting.
- Claude should keep the current 2-4 track demo set playable and choose a new set when the set is exhausted or the genre/cluster needs to change.
- Track-boundary playback should not wait on Claude, embedding search, Redis, or TTS. It should execute a ready transition plan or continue with deterministic fallback playback.
- Redis provides compact context so Claude does not need the full event history.

## Success criteria

- The demo starts autonomously without requiring user input.
- The mascot app surface starts cleanly and can later show current-song context clearly.
- The system starts with a coherent 2-4 track demo set.
- Reaction changes cause visible queue updates.
- Positive feedback leads to similar songs.
- Negative feedback shifts away from the current music cluster.
- Redis is clearly used beyond caching.
- The pitch can explain the system in one sentence: a Claude-driven DJ agent that uses Redis memory and vector search to adapt music from live reactions.

## Non-goals

- No chat/request flow required for the demo.
- No full chat interface.
- No manual queue editor.
- No lyrics requirement for tracks in the recommendation pool.
- No claim that Spotify provides native song embeddings.
- No requirement to use Redis Iris as the center if structured Redis state is sufficient.
