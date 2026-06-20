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

`uv run claude-dj` starts a long-running autonomous process. There is no chat prompt and no user request trigger. `Ctrl+C` stops the process.

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

These tools intentionally return stub data for now. They should be replaced internally as other workstreams land, without changing Claude's tool contract.

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
uv run claude-dj
```

The test suite uses fake SDK clients and fake boundary adapters so it does not require live Claude, Redis, Spotify, Deepgram, or camera/audio workers.

## Integration Notes

Boundary execution must stay deterministic. Claude may prepare decisions mid-song, but the track boundary must not wait on Claude, Redis search, embedding search, or Deepgram TTS.

If a prepared narration asset is not ready at the boundary, playback should continue with a deterministic fallback rather than pausing.
