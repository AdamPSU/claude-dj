---
name: on-queue-refresh
kind: hook
hook: on_queue_refresh
---

<hook>
  on_queue_refresh
</hook>

<context>
  The current song is still playing, but ClaudeDJ's app-owned queue has no upcoming tracks.
  Prepare the next set in the background without interrupting playback.
</context>

<playback>
  {playback_json}
</playback>

<workflow>
  Call get_current_playback.
  Call get_session_context.
  Call search_track_embeddings with seed_track_id="{current_track_id}", mode="similar", signal="positive", exclude_recent=true, and limit=6.
  Choose exactly 1-2 candidates that continue the current direction unless session context clearly says to shift.
  Call replace_queue with timing="after_current_track" and reason="same_lane_refill" for same-direction refills.
  Only call narrate with mode="prepare" if changing direction; do not narrate for a same-lane refill.
</workflow>

<constraint>
  Do not call play_track.
  Do not stop, pause, skip, or interrupt the current song.
</constraint>
