from __future__ import annotations

import json
from importlib.resources import files


def load_prompt(filename: str) -> str:
    return files(__package__).joinpath(filename).read_text(encoding="utf-8")


DJ_SYSTEM_PROMPT = load_prompt("system.md")
START_HOOK_PROMPT = load_prompt("on_start.md")


def build_reaction_event_prompt(event: dict[str, object]) -> str:
    current_cluster = event.get("current_cluster")
    avoid_clusters = [current_cluster] if current_cluster else []
    event_type = str(event.get("event_type", "reaction_event"))
    search_signal = "neutral" if event_type == "max_cluster_streak_reached" else "negative"
    return load_prompt("on_reaction_event.md").format(
        event_json=json.dumps(event, sort_keys=True),
        event_type=event_type,
        search_signal=search_signal,
        current_track_id=event.get("current_track_id", ""),
        avoid_clusters=json.dumps(avoid_clusters),
    )


def build_queue_refresh_prompt(playback: dict[str, object]) -> str:
    return load_prompt("on_queue_refresh.md").format(
        playback_json=json.dumps(playback, sort_keys=True),
        current_track_id=playback.get("current_track_id", ""),
    )
