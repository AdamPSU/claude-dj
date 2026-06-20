---
name: on-mid-song-prepare
kind: hook
hook: on_mid_song_prepare
---

<hook>
  on_mid_song_prepare
</hook>

<context>
  The current song is still playing at {progress_percent}% progress.
  Prepare in the background only.
</context>

<workflow>
  Call get_current_playback.
  Call get_session_context.
  Call get_reaction_signal.
  If the current genre or cluster is working, do nothing else.
  If the current genre or cluster is not working, call search_track_embeddings for a shifted direction.
  If shifting, call replace_queue with timing="after_current_track" and a 3-6 song set.
  If shifting, call narrate with mode="prepare", timing="after_current_track", current_track_id, next_track_id, and track_ids so the boundary can play prepared narration instantly.
</workflow>

<constraint>
  Do not stop, pause, skip, or interrupt the current song.
</constraint>
