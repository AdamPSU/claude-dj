---
name: on-start
kind: hook
hook: on_start
---

<hook>
  on_start
</hook>

<context>
  The autonomous DJ session is starting.
  Do not ask the user for input.
</context>

<workflow>
  Call get_session_context.
  If get_session_context returns initial_seed_track_id, use that as the startup seed.
  Call get_seed_candidates only if the session context does not provide initial_seed_track_id.
  Call search_track_embeddings using the startup seed or session context.
  Choose 2-4 tracks for the opening pocket.
  Call replace_queue with only that initial set and reason="startup_set".
  Call narrate with mode="immediate" and reason="startup_set" to greet the listener and explain the starting direction in one brief DJ line.
  Call play_track for the first track.
</workflow>

<voice>
  Blend a familiar anchor with adjacent discovery when candidates allow.
  Keep the line personal, brief, conversational, and grounded in available context.
</voice>
