# ClaudeDJ

ClaudeDJ is an autonomous DJ harness for the Berkeley AI Hackathon. Claude Code SDK runs the DJ agent, a project MCP server exposes music tools, and playback transitions are executed deterministically so the user does not wait on Claude at song boundaries.

## Current Scope

This branch owns the harness and MCP integration surface.

- `claude_dj/main.py`: script entrypoint for the long-running process.
- `claude_dj/agent/`: Claude SDK lifecycle code.
- `claude_dj/agent/prompts/`: markdown prompts with YAML frontmatter and simple XML sections.
- `claude_dj/mcp/`: project DJ MCP tool handlers and SDK MCP server registration.
- `claude_dj/transition.py`: deterministic track-boundary transition execution.

Other teammates own Redis, embeddings, retrieval internals, realtime reaction detection, face analysis, and sound analysis.

## Runtime Model

`uv run python -m claude_dj.main` starts the long-running Claude-driven process. There is no chat prompt and no user request trigger. `Ctrl+C` stops the process.

By default the long-running harness prints only lifecycle-level messages. For demo debugging, enable the full Claude SDK stream with `--verbose-claude` or `CLAUDE_DJ_VERBOSE_CLAUDE=1`; this prints message type, assistant text, MCP tool calls, MCP tool results, rate-limit status, and final result status for each turn.

`uv run python -m claude_dj` starts the bounded user-facing demo. It infers a starting direction from Spotify context, narrates the choice, plays the narration locally, starts Spotify playback, and exits after confirming the pipeline is live.

The harness has three key lifecycle points:

- `on_start`: Claude wakes once, reads session context, searches tracks, chooses a 3-6 song set, narrates the start, and starts playback through MCP tools.
- `on_mid_song_prepare`: Claude wakes in the background while the current song is still playing. If reactions say the current direction is not working, Claude prepares a shifted song set and pre-renders bridge narration.
- `on_track_boundary`: no Claude call. The boundary executor only uses a ready transition plan or does nothing.

## Key Contracts

`ClaudeDJ`

Wraps the Claude Code SDK client. It builds `ClaudeAgentOptions`, registers the in-process DJ MCP server, restricts allowed tools, sends hook prompts, and drains Claude responses.

`DJAgentRunner`

Routes lifecycle hooks. It sends `on_start` and `on_mid_song_prepare` to `ClaudeDJ`, but sends `on_track_boundary` to playback transition execution. This keeps Claude off the realtime boundary path.

`DJToolHandlers`

Implements the project MCP tool surface. The handlers are stubs right now, but their names and shapes are the integration contract for Redis, embeddings, playback, Deepgram, and reaction workers.

`mcp/playback.py`

Owns ClaudeDJ's app-side playback state: current track, active queue, pending queue, recent tracks, and cluster streak. `replace_queue`, `play_track`, and `get_current_playback` use this state so Spotify's native queue does not become the source of truth.

`mcp/spotify.py`

Wraps Spotify Web API calls. It refreshes an access token from `SPOTIFY_REFRESH_TOKEN`, starts tracks with `PUT /v1/me/player/play`, reads current playback with `GET /v1/me/player`, searches tracks with `GET /v1/search`, and reads the user's playlists for temporary recommendation candidates.

`mcp/narration.py`

Implements the Deepgram-backed narration path for the MCP `narrate` tool. Generated audio is short-lived and stored in memory by narration id for startup or prepared transition playback.

`BoundaryExecutor`

Executes ready transitions without Claude. It validates the prepared plan against the ending track, ducks music volume to 10%, plays the prepared next track and narration, then restores the previous volume.

`TransitionPlan`

Stores the prepared bridge:

```python
current_track_id
next_track_id
track_ids
narration_id
ready
```

## MCP Tools

The in-process MCP server is named `dj` and currently exposes:

- `get_session_context`
- `search_track_embeddings`
- `replace_queue`
- `narrate`
- `play_track`
- `get_current_playback`
- `get_reaction_signal`
- `mark_track_feedback`
- `summarize_session`
- `search_session_history`

Playback tools now use app-owned queue state plus Spotify playback control. `search_track_embeddings` temporarily searches the user's Spotify playlists first, then Spotify global search, and returns those candidates through the existing embedding-search contract until Redis/CLAP retrieval lands. Reaction, feedback, and history internals still use demo/stub behavior until the Redis and reaction workstreams land.

## Prompt Files

Prompts live in `src/backend/claude_dj/agent/prompts/`.

- `system.md`: ClaudeDJ identity, autonomy rules, MCP boundary, queue policy.
- `on_start.md`: startup workflow.
- `on_mid_song_prepare.md`: mid-song background preparation workflow.

Prompt format:

```markdown
---
name: example
kind: hook
---

<section>
  Plain text instructions.
  No nested XML item tags.
</section>
```

`on_mid_song_prepare.md` is a template. `build_mid_song_prompt(progress_percent=...)` loads it and fills `{progress_percent}`.

## Development

From `src/backend`:

```bash
uv run python -m unittest discover -s tests
uv run python -m compileall claude_dj tests
uv run python -m claude_dj
uv run python -m claude_dj.main
uv run python -m claude_dj.main --verbose-claude
```

Required runtime env for Spotify playback:

```bash
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=
SPOTIFY_REFRESH_TOKEN=
CLAUDE_DJ_DEMO_TRACK_URIS=spotify:track:...,spotify:track:...
```

Recommended local OAuth redirect URI for procuring the refresh token:

```text
http://127.0.0.1:8888/callback
```

The redirect URI must match the Spotify Developer Dashboard value exactly. Current Spotify docs require explicit loopback IP addresses for local redirect URIs, such as `http://127.0.0.1:<PORT>/callback`, rather than `localhost`.

Required Spotify scopes for the refresh token:

```text
user-read-playback-state
user-modify-playback-state
streaming
playlist-read-private
playlist-read-collaborative
```

ClaudeDJ omits Spotify's optional `device_id` in the playback request. The playback runtime still handles Spotify Connect activation: before `play_track`, it checks current playback for an active unrestricted device, lists devices if needed, transfers to the remembered or first unrestricted device, then starts the selected Spotify URI. Keep Spotify desktop, mobile, or web open before starting the harness so there is at least one available device.

`uv run python -m claude_dj` is the bounded live user demo. It loads `.env`, activates an available Spotify Connect device if needed, infers a starting direction from current Spotify context and playlist names, searches candidates through the same runtime path, generates a Deepgram narration line, plays that narration locally with macOS `afplay`, starts Spotify playback, then reads current playback back from Spotify. Do not pass a genre for the real demo; an optional `--query` override exists only for manual experiments. Expected final line:

```text
demo: ok
```

The test suite uses fake SDK clients and fake boundary adapters so it does not require live Claude, Redis, Spotify, Deepgram, or camera/audio workers.

## Integration Notes

Boundary execution must stay deterministic. Claude may prepare decisions mid-song, but the track boundary must not wait on Claude, Redis search, embedding search, or Deepgram TTS.

If a prepared narration asset is not ready at the boundary, playback should continue with a deterministic fallback rather than pausing.
