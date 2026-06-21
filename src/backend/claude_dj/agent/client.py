from __future__ import annotations

import os
import sys
from typing import Any
from typing import TextIO

import sentry_sdk

from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
)

from ..mcp.handlers import ReactionSource
from ..mcp.server import build_dj_mcp_server
from ..mcp.narration import NarrationPlayer, Narrator
from ..mcp.playback import InMemoryPlaybackRuntime
from ..observability import observe_async
from ..transition import InMemoryTransitionStore
from .prompts import (
    DJ_SYSTEM_PROMPT,
    START_HOOK_PROMPT,
    build_queue_refresh_prompt,
    build_reaction_event_prompt,
)


CLAUDE_DJ_MODEL = "claude-opus-4-7"
CLAUDE_DJ_EFFORT = "medium"


def build_allowed_tools() -> list[str]:
    return [
        "mcp__dj__get_session_context",
        "mcp__dj__search_track_embeddings",
        "mcp__dj__get_seed_candidates",
        "mcp__dj__replace_queue",
        "mcp__dj__narrate",
        "mcp__dj__play_track",
        "mcp__dj__get_current_playback",
        "mcp__dj__get_reaction_signal",
        "mcp__dj__mark_track_feedback",
        "mcp__dj__summarize_session",
        "mcp__dj__search_session_history",
    ]


def build_extra_args() -> dict[str, str | None]:
    if os.environ.get("CLAUDE_DJ_CLAUDE_FAST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return {"bare": None}
    return {}


def build_agent_options(
    store: InMemoryTransitionStore,
    narrator: Narrator,
    playback: InMemoryPlaybackRuntime | None = None,
    narration_player: NarrationPlayer | None = None,
    reaction_source: ReactionSource | None = None,
) -> Any:
    try:
        from claude_agent_sdk import ClaudeAgentOptions
    except ImportError as exc:
        raise RuntimeError(
            "claude-agent-sdk is required to run ClaudeDJ. Install backend dependencies with uv first."
        ) from exc

    return ClaudeAgentOptions(
        tools=[],
        system_prompt=DJ_SYSTEM_PROMPT,
        mcp_servers={"dj": build_dj_mcp_server(store, narrator, playback, narration_player, reaction_source)},
        allowed_tools=build_allowed_tools(),
        strict_mcp_config=True,
        max_turns=8,
        model=CLAUDE_DJ_MODEL,
        effort=CLAUDE_DJ_EFFORT,
        extra_args=build_extra_args(),
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
    def __init__(
        self,
        *,
        client: Any | None = None,
        options: Any | None = None,
        output: TextIO = sys.stdout,
        verbose_claude: bool = False,
    ) -> None:
        self.options = options
        self.client = client if client is not None else build_sdk_client(options)
        self._connected = False
        self.output = output
        self.verbose_claude = verbose_claude

    @classmethod
    def create(
        cls,
        store: InMemoryTransitionStore,
        narrator: Narrator,
        playback: InMemoryPlaybackRuntime | None = None,
        narration_player: NarrationPlayer | None = None,
        reaction_source: ReactionSource | None = None,
        output: TextIO = sys.stdout,
        verbose_claude: bool = False,
    ) -> ClaudeDJ:
        options = build_agent_options(store, narrator, playback, narration_player, reaction_source)
        return cls(options=options, output=output, verbose_claude=verbose_claude)

    async def connect(self) -> None:
        if self._connected:
            return
        await self.client.connect()
        self._connected = True

    async def disconnect(self) -> None:
        if not self._connected:
            return
        await self.client.disconnect()
        self._connected = False

    async def handle_start(self) -> None:
        await observe_async(
            "claude_dj.agent.on_start",
            op="claude_dj.agent",
            data={"hook": "on_start"},
            callback=lambda: self._send_turn(START_HOOK_PROMPT),
        )

    async def handle_reaction_event(self, event: dict[str, object]) -> None:
        await observe_async(
            "claude_dj.agent.on_reaction_event",
            op="claude_dj.agent",
            data={"hook": "on_reaction_event", "event_type": str(event.get("event_type", "reaction_event"))},
            callback=lambda: self._send_turn(build_reaction_event_prompt(event)),
        )

    async def handle_queue_refresh(self, playback: dict[str, object]) -> None:
        await observe_async(
            "claude_dj.agent.on_queue_refresh",
            op="claude_dj.agent",
            data={"hook": "on_queue_refresh", "current_track_id": str(playback.get("current_track_id", ""))},
            callback=lambda: self._send_turn(build_queue_refresh_prompt(playback)),
        )

    async def _send_turn(self, prompt: str) -> None:
        with sentry_sdk.start_span(op="claude_agent_sdk.query", name="ClaudeSDKClient.query") as span:
            span.set_data("prompt_chars", len(prompt))
            await self.client.query(prompt)

        message_count = 0
        error_result: ResultMessage | None = None
        with sentry_sdk.start_span(op="claude_agent_sdk.receive", name="ClaudeSDKClient.receive_response") as span:
            async for message in self.client.receive_response():
                message_count += 1
                if self.verbose_claude:
                    self._log_message(message)
                if isinstance(message, ResultMessage) and message.is_error:
                    error_result = message
            span.set_data("message_count", message_count)
            span.set_data("is_error", bool(error_result))
        if error_result is not None:
            raise RuntimeError(f"Claude turn failed: {error_result.errors or error_result.result or error_result.stop_reason}")

    def _log_message(self, message: object) -> None:
        print(f"claude: message {type(message).__name__}", file=self.output, flush=True)
        if isinstance(message, AssistantMessage):
            print(
                f"claude: assistant stop_reason={message.stop_reason} error={message.error}",
                file=self.output,
                flush=True,
            )
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"claude: assistant text {block.text}", file=self.output, flush=True)
                elif isinstance(block, ToolUseBlock):
                    print(f"claude: tool_use {block.name} {block.input}", file=self.output, flush=True)
                else:
                    print(f"claude: assistant block {block}", file=self.output, flush=True)
            return

        if isinstance(message, UserMessage):
            print(f"claude: user content {message.content}", file=self.output, flush=True)
            print(f"claude: tool_result {message.tool_use_result}", file=self.output, flush=True)
            return

        if isinstance(message, SystemMessage):
            print(
                f"claude: system subtype={message.subtype} data={message.data}",
                file=self.output,
                flush=True,
            )
            return

        if isinstance(message, ResultMessage):
            print(
                "claude: result "
                f"subtype={message.subtype} is_error={message.is_error} "
                f"stop_reason={message.stop_reason} errors={message.errors} result={message.result}",
                file=self.output,
                flush=True,
            )
            return

        if isinstance(message, RateLimitEvent):
            info = message.rate_limit_info
            print(
                "claude: rate_limit "
                f"status={info.status} type={info.rate_limit_type} "
                f"resets_at={info.resets_at} overage_status={info.overage_status} "
                f"overage_disabled_reason={info.overage_disabled_reason}",
                file=self.output,
                flush=True,
            )
            return

        print(f"claude: raw {message}", file=self.output, flush=True)
