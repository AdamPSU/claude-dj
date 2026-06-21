# ClaudeDJ example

This example shows one full autonomous ClaudeDJ lifecycle. It first gives the high-level session, then shows the same scenario from Claude's lower-level tool view.

## Example: Autonomous reggaeton seed, mostly positive feedback

Startup context:

- configured seed vibe: reggaeton
- no strong recent dislikes
- yesterday leaned indie pop and synthwave
- startup rule: choose one coherent 2-4 track demo set; do not extend it immediately
- bridge rule: narrate when switching genres/clusters because the current direction did not land

High-level lifecycle:

- Song 1 starts from the initial reggaeton set. Claude narrates the start, ideally as a short Deepgram TTS line plus mini-player status.
- Song 2 stays in the same set. No narration unless the direction changes.
- Song 3 stays close. Signal is positive. Minimum run is satisfied at `3/6`.
- Song 4 stays close but adds slight novelty. Signal is neutral-positive.
- Song 5 keeps the groove. Signal is positive.
- Song 2-4 reaches the selected group target. Claude prepares an adjacent shift or freshens the pocket.
- Song 7 shifts to nearby Latin pop / dancehall. Claude narrates the genre bridge.
- Signal is positive. Claude treats this as the new working cluster.
- Song 8 stays near the new cluster.
- Song 9 gets neutral feedback. Claude keeps the queue stable because the cluster is still young.
- Song 10 gets positive feedback. Claude refreshes similar tracks.
- Song 11 gets weak feedback. Claude prepares a slight shift.
- Song 12 shifts toward smoother, lower-energy tracks.
- Songs 13-15 continue in the smoother pocket unless feedback turns negative.

End state:

- liked clusters: reggaeton, Latin pop / dancehall
- avoided clusters: none
- session summary: user responded well to rhythmic Latin tracks; slight novelty worked after 6-song run
- mascot app mostly stayed ambient; future playback/status text can show the current track and short bridge line when the genre changes

## Low-level Claude view

Startup hook:

```text
hook: on_start
```

Claude sees:

```json
{
  "session": {
    "seed_vibe": "reggaeton",
    "recent_dislikes": [],
    "yesterday_genres": ["indie pop", "synthwave"],
    "cluster_policy": {"min": 3, "max": 6}
  }
}
```

Claude calls:

```text
search_track_embeddings({
  "query": "reggaeton",
  "mode": "text",
  "limit": 12,
  "exclude_recent": true
})
```

Tool returns:

```json
{
  "candidates": [
    {"id": "t1", "title": "Track 1", "cluster": "reggaeton_a", "score": 0.91},
    {"id": "t2", "title": "Track 2", "cluster": "reggaeton_a", "score": 0.88}
  ]
}
```

Claude calls:

```text
replace_queue({"track_ids": ["t1", "t2"], "reason": "startup_set"})
narrate({"text": "I found a reggaeton pocket. I will start here and adjust as I read the room."})
play_track({"track_id": "t1"})
```

The `narrate` tool should return immediately with display text and generate Deepgram TTS audio when configured.

After song 1 midpoint, Claude sees:

```json
{
  "current_track": {"id": "t1", "cluster": "reggaeton_a"},
  "reaction_score": 0.78,
  "seconds_remaining": 92,
  "cluster_streak": 1,
  "queue": ["t2"]
}
```

Claude calls:

```text
mark_track_feedback({"track_id": "t1", "feedback": "liked", "score": 0.78})
search_track_embeddings({
  "seed_track_id": "t1",
  "mode": "similar",
  "limit": 6,
  "exclude_recent": true
})
replace_queue({"track_ids": ["t2", "t3", "t4", "t5", "t6", "t7"]})
```

After songs 2-5, Claude repeatedly sees positive or neutral-positive scores:

```json
{
  "current_cluster": "reggaeton_a",
  "cluster_streak": 5,
  "recent_feedback": [
    {"track_id": "t1", "score": 0.78},
    {"track_id": "t2", "score": 0.71},
    {"track_id": "t3", "score": 0.75},
    {"track_id": "t4", "score": 0.43},
    {"track_id": "t5", "score": 0.68}
  ],
  "recommended_action": "stay_close_until_max"
}
```

Claude does not narrate. It keeps the current set going.

At song 6, Claude sees:

```json
{
  "current_track": {"id": "t6", "cluster": "reggaeton_a"},
  "reaction_score": 0.66,
  "cluster_streak": 6,
  "cluster_policy": {"min": 3, "max": 6},
  "recommended_action": "shift_adjacent"
}
```

Claude calls:

```text
search_track_embeddings({
  "seed_track_id": "t6",
  "mode": "adjacent_shift",
  "limit": 6,
  "avoid_clusters": ["reggaeton_a"]
})
```

Tool returns:

```json
{
  "candidates": [
    {"id": "t7", "title": "Track 7", "cluster": "latin_pop_dancehall", "score": 0.79},
    {"id": "t8", "title": "Track 8", "cluster": "latin_pop_dancehall", "score": 0.76},
    {"id": "t9", "title": "Track 9", "cluster": "latin_pop_dancehall", "score": 0.74}
  ]
}
```

Claude calls:

```text
replace_queue({"track_ids": ["t7", "t8", "t9"]})
narrate({"text": "This groove is working, so I am moving one step over instead of snapping away."})
```

After song 7 midpoint, Claude sees:

```json
{
  "current_track": {"id": "t7", "cluster": "latin_pop_dancehall"},
  "reaction_score": 0.72,
  "cluster_streak": 1,
  "recommended_action": "treat_as_new_positive_cluster"
}
```

Claude calls:

```text
mark_track_feedback({"track_id": "t7", "feedback": "liked", "score": 0.72})
search_track_embeddings({
  "seed_track_id": "t7",
  "mode": "similar",
  "limit": 6,
  "exclude_recent": true
})
replace_queue({"track_ids": ["t8", "t9", "t10", "t11", "t12", "t13"]})
```

After song 11, Claude sees weak feedback:

```json
{
  "current_track": {"id": "t11", "cluster": "latin_pop_dancehall"},
  "reaction_score": 0.18,
  "cluster_streak": 5,
  "recommended_action": "prepare_slight_shift"
}
```

Claude calls:

```text
search_track_embeddings({
  "seed_track_id": "t11",
  "mode": "slight_shift",
  "limit": 6,
  "exclude_recent": true
})
replace_queue({"track_ids": ["t12", "t13", "t14", "t15"]})
```

At session end, Claude calls:

```text
summarize_session({
  "liked_clusters": ["reggaeton_a", "latin_pop_dancehall"],
  "disliked_clusters": [],
  "notes": "User responded well to rhythmic Latin tracks. Adjacent shift after six-song run preserved engagement."
})
```
