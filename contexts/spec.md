# DJ agent product spec

## Summary

Build a minimal, emotion-aware DJ that keeps music playing, watches user reactions, and adjusts the queue using CLAP audio embeddings plus Spotify metadata.

The user sees a small draggable mini player. The system behind it is an agent harness: Claude Code SDK drives decisions, our MCP server exposes music tools, and Redis stores memory, vectors, state, and retrieval context.

## Core loop

The user hears music.
The backend watches reactions.
Redis remembers what happened.
Vector search finds possible next songs.
The DJ ranks them using recent context.
Then the player plays the next song.

## User experience

- User starts with a simple request, such as "play reggaeton."
- A small draggable mini player appears, like a Spotify mini player.
- The mini player shows album art, title, artist, and one short status line.
- No skip button, queue editor, or large dashboard.
- The DJ may narrate short transitions, especially when it starts or changes direction.
- The system should feel ambient, not like a chat app.

## Primary behavior

- Claude searches track embeddings for the requested vibe.
- Claude queues the top 3-6 tracks.
- Claude narrates the starting choice.
- Playback begins.
- Mid-song reaction signals are collected.
- If the user seems to like the song, Claude refreshes the queue with similar tracks.
- If the user seems not to like the song, Claude marks the current music cluster as disliked, replaces the queue with shifted candidates, and narrates the change.
- If the signal is neutral, the DJ can make a slight shift after the minimum run is satisfied.

## Similarity run rule

- Stay in a working music cluster for at least 3 songs.
- Leave a music cluster after 6 songs to avoid staleness.
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
- Memory/context records: store recent songs, liked clusters, disliked clusters, yesterday's genres, and cluster streaks.
- Session history: store searchable listening summaries, played tracks, reactions, and time ranges.

## Claude / agent usage

- Claude Code SDK is the agent runtime.
- A custom DJ mission prompt tells Claude how to manage the queue.
- Our MCP server gives Claude tools for playback, retrieval, memory, and narration.
- Claude should not wait for a song to end before acting.
- Claude should keep a queue ready and update it during playback.
- Redis provides compact context so Claude does not need the full event history.

## Success criteria

- The demo starts from a natural music request.
- The mini player shows the current song clearly.
- The system queues multiple tracks ahead.
- Reaction changes cause visible queue updates.
- Positive feedback leads to similar songs.
- Negative feedback shifts away from the current music cluster.
- Redis is clearly used beyond caching.
- The pitch can explain the system in one sentence: a Claude-driven DJ agent that uses Redis memory and vector search to adapt music from live reactions.

## Non-goals

- No full chat interface.
- No manual queue editor.
- No lyrics requirement for tracks in the recommendation pool.
- No claim that Spotify provides native song embeddings.
- No requirement to use Redis Iris as the center if structured Redis state is sufficient.
