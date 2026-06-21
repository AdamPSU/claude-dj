---
name: on-reaction-event
kind: hook
hook: on_reaction_event
---

<hook>
  on_reaction_event
</hook>

<context>
  The reaction monitor detected an actionable listener reaction while the current song is still playing.
  The event was already thresholded outside Claude, so do not re-poll for several neutral samples.
</context>

<event>
  {event_json}
</event>

<workflow>
  Call get_current_playback.
  Call get_session_context.
  Call get_reaction_signal once to read the latest compact signal.
  If this event is sustained_negative_reaction, call mark_track_feedback for the current track with feedback="disliked" before planning the shift.
  If this event is max_cluster_streak_reached, do not mark negative feedback; this is freshening the set, not listener rejection.
  Because this event is {event_type}, call search_track_embeddings with seed_track_id="{current_track_id}", mode="shift", signal="{search_signal}", avoid_clusters={avoid_clusters}, exclude_recent=true, and limit=6.
  Choose 2-4 shifted candidates for the next pocket.
  Call replace_queue with timing="after_current_track" and reason="reaction_shift" for sustained negative reactions or reason="cluster_rotation" for cluster policy.
  Call narrate with mode="prepare", timing="after_current_track", reason="reaction_shift" or reason="cluster_rotation", current_track_id, next_track_id, and track_ids so the boundary can play prepared narration instantly.
</workflow>

<constraint>
  Do not call play_track.
  Do not stop, pause, skip, or interrupt the current song.
  Do not blame the listener or mention camera analysis; describe the musical pivot.
</constraint>
