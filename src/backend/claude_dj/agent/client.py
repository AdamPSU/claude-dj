from __future__ import annotations

from typing import Any

import sentry_sdk

from ..mcp.server import build_dj_mcp_server
from ..mcp.narration import Narrator
from ..observability import observe_async
from ..transition import InMemoryTransitionStore
from .prompts import DJ_SYSTEM_PROMPT, START_HOOK_PROMPT, build_mid_song_prompt


def build_allowed_tools() -> list[str]:
    return [
        "mcp__dj__get_session_context",
        "mcp__dj__search_track_embeddings",
        "mcp__dj__replace_queue",
        "mcp__dj__narrate",
        "mcp__dj__play_track",
        "mcp__dj__get_current_playback",
        "mcp__dj__get_reaction_signal",
        "mcp__dj__mark_track_feedback",
        "mcp__dj__summarize_session",
        "mcp__dj__search_session_history",
    ]


def build_agent_options(store: InMemoryTransitionStore, narrator: Narrator) -> Any:
    try:
        from claude_agent_sdk import ClaudeAgentOptions
    except ImportError as exc:
        raise RuntimeError(
            "claude-agent-sdk is required to run ClaudeDJ. Install backend dependencies with uv first."
        ) from exc

    return ClaudeAgentOptions(
        system_prompt=DJ_SYSTEM_PROMPT,
        mcp_servers={"dj": build_dj_mcp_server(store, narrator)},
        allowed_tools=build_allowed_tools(),
        max_turns=8,
    )


def build_sdk_client(options: Any) -> Any:
    try:
        from claude_agent_sdk import ClaudeSDKClient
    except ImportError as exc:
        raise RuntimeError(
            "claude-agent-sdk is required to run ClaudeDJ. Install backend dependencies with uv first."
        ) from exc

    return ClaudeSDKClient(options=options)


class ClaudeDJ:
    def __init__(self, *, client: Any | None = None, options: Any | None = None) -> None:
        self.options = options
        self.client = client if client is not None else build_sdk_client(options)

    @classmethod
    def create(cls, store: InMemoryTransitionStore, narrator: Narrator) -> ClaudeDJ:
        options = build_agent_options(store, narrator)
        return cls(options=options)

    async def handle_start(self) -> None:
        await observe_async(
            "claude_dj.agent.on_start",
            op="claude_dj.agent",
            data={"hook": "on_start"},
            callback=lambda: self._send_turn(START_HOOK_PROMPT),
        )

    async def handle_mid_song_prepare(self, progress_percent: int) -> None:
        await observe_async(
            "claude_dj.agent.on_mid_song_prepare",
            op="claude_dj.agent",
            data={"hook": "on_mid_song_prepare", "progress_percent": progress_percent},
            callback=lambda: self._send_turn(build_mid_song_prompt(progress_percent=progress_percent)),
        )

    async def _send_turn(self, prompt: str) -> None:
        with sentry_sdk.start_span(op="claude_agent_sdk.query", name="ClaudeSDKClient.query") as span:
            span.set_data("prompt_chars", len(prompt))
            await self.client.query(prompt)

        message_count = 0
        with sentry_sdk.start_span(op="claude_agent_sdk.receive", name="ClaudeSDKClient.receive_response") as span:
            async for _message in self.client.receive_response():
                message_count += 1
            span.set_data("message_count", message_count)
