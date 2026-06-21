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
  Call get_seed_candidates to find a grounded Redis seed if the session does not already provide one.
  Call search_track_embeddings using the seed or session context.
  Choose exactly 1-2 tracks.
  Call replace_queue with only that initial set.
  Call narrate with mode="immediate" to greet the listener and explain the starting direction.
  Call play_track for the first track.
</workflow>
