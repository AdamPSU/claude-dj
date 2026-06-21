from __future__ import annotations

import json
from typing import Any

from ..transition import InMemoryTransitionStore
from .handlers import DJToolHandlers, ReactionSource
from .narration import NarrationPlayer, Narrator
from .playback import InMemoryPlaybackRuntime


def mcp_json_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def build_dj_mcp_server(
    transition_store: InMemoryTransitionStore,
    narrator: Narrator,
    playback: InMemoryPlaybackRuntime | None = None,
    narration_player: NarrationPlayer | None = None,
    reaction_source: ReactionSource | None = None,
) -> Any:
    try:
        from claude_agent_sdk import create_sdk_mcp_server, tool
    except ImportError as exc:
        raise RuntimeError(
            "claude-agent-sdk is required to build the ClaudeDJ MCP server."
        ) from exc

    handlers = DJToolHandlers(transition_store, narrator, playback, narration_player, reaction_source)

    @tool("get_session_context", "Return compact ClaudeDJ session context.", {"type": "object", "properties": {}})
    async def get_session_context(_input: dict[str, Any]) -> dict[str, Any]:
        return mcp_json_result(await handlers.get_session_context())

    @tool(
        "search_track_embeddings",
        "Search Redis-backed track embeddings from a seed track.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {"type": "string"},
                "seed_track_id": {"type": "string"},
                "signal": {"type": "string"},
                "avoid_clusters": {"type": "array", "items": {"type": "string"}},
                "exclude_recent": {"type": "boolean"},
                "exclude_track_ids": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer"},
            },
        },
    )
    async def search_track_embeddings(input: dict[str, Any]) -> dict[str, Any]:
        return mcp_json_result(await handlers.search_track_embeddings(**input))

    @tool(
        "get_seed_candidates",
        "Return Redis-backed starter tracks Claude may choose as a seed.",
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "avoid_clusters": {"type": "array", "items": {"type": "string"}},
            },
        },
    )
    async def get_seed_candidates(input: dict[str, Any]) -> dict[str, Any]:
        return mcp_json_result(await handlers.get_seed_candidates(**input))

    @tool(
        "replace_queue",
        "Replace the current or pending ClaudeDJ track set.",
        {
            "type": "object",
            "properties": {
                "track_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
                "timing": {"type": "string"},
            },
            "required": ["track_ids", "reason"],
        },
    )
    async def replace_queue(input: dict[str, Any]) -> dict[str, Any]:
        return mcp_json_result(await handlers.replace_queue(**input))

    @tool(
        "narrate",
        "Display or prepare a short DJ narration line.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "mode": {"type": "string"},
                "reason": {"type": "string"},
                "timing": {"type": "string"},
                "current_track_id": {"type": "string"},
                "next_track_id": {"type": "string"},
                "track_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text", "mode", "reason"],
        },
    )
    async def narrate(input: dict[str, Any]) -> dict[str, Any]:
        return mcp_json_result(await handlers.narrate(**input))

    @tool(
        "play_track",
        "Start a track by id.",
        {"type": "object", "properties": {"track_id": {"type": "string"}}, "required": ["track_id"]},
    )
    async def play_track(input: dict[str, Any]) -> dict[str, Any]:
        return mcp_json_result(await handlers.play_track(**input))

    @tool("get_current_playback", "Return current playback state.", {"type": "object", "properties": {}})
    async def get_current_playback(_input: dict[str, Any]) -> dict[str, Any]:
        return mcp_json_result(await handlers.get_current_playback())

    @tool("get_reaction_signal", "Return recent reaction signal.", {"type": "object", "properties": {}})
    async def get_reaction_signal(_input: dict[str, Any]) -> dict[str, Any]:
        return mcp_json_result(await handlers.get_reaction_signal())

    @tool(
        "mark_track_feedback",
        "Persist track or cluster feedback. Stubbed until Redis integration is ready.",
        {
            "type": "object",
            "properties": {
                "track_id": {"type": "string"},
                "feedback": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["track_id", "feedback"],
        },
    )
    async def mark_track_feedback(input: dict[str, Any]) -> dict[str, Any]:
        return mcp_json_result(await handlers.mark_track_feedback(**input))

    @tool(
        "summarize_session",
        "Write a compact session summary. Stubbed until Redis integration is ready.",
        {"type": "object", "properties": {"summary": {"type": "object"}}},
    )
    async def summarize_session(input: dict[str, Any]) -> dict[str, Any]:
        return mcp_json_result(await handlers.summarize_session(**input))

    @tool(
        "search_session_history",
        "Search previous sessions. Stubbed until Redis integration is ready.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    )
    async def search_session_history(input: dict[str, Any]) -> dict[str, Any]:
        return mcp_json_result(await handlers.search_session_history(**input))

    return create_sdk_mcp_server(
        name="dj",
        version="0.1.0",
        tools=[
            get_session_context,
            search_track_embeddings,
            get_seed_candidates,
            replace_queue,
            narrate,
            play_track,
            get_current_playback,
            get_reaction_signal,
            mark_track_feedback,
            summarize_session,
            search_session_history,
        ],
    )
