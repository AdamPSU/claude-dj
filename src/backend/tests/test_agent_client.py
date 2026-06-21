import io
import os
import unittest
from unittest.mock import patch

from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from claude_dj.agent.client import ClaudeDJ, build_agent_options, build_allowed_tools
from claude_dj.transition import InMemoryTransitionStore


class FakeSDKClient:
    def __init__(self, messages: list[object] | None = None) -> None:
        self.connected = False
        self.events: list[str] = []
        self.messages = messages or []

    async def connect(self) -> None:
        self.connected = True
        self.events.append("connect")

    async def disconnect(self) -> None:
        self.connected = False
        self.events.append("disconnect")

    async def query(self, prompt: str) -> None:
        if not self.connected:
            raise RuntimeError("query before connect")
        self.events.append("query")

    async def receive_response(self):
        if not self.connected:
            raise RuntimeError("receive before connect")
        self.events.append("receive")
        for message in self.messages:
            yield message


class ClaudeDJClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_options_default_to_oauth_safe_mode_without_lowering_effort(self) -> None:
        with patch("claude_dj.agent.client.build_dj_mcp_server", return_value={"type": "sdk"}), patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            options = build_agent_options(InMemoryTransitionStore(), narrator=object())

        self.assertEqual(options.model, "claude-opus-4-7")
        self.assertEqual(options.effort, "medium")
        self.assertEqual(options.extra_args, {})

    async def test_agent_options_enable_bare_fast_mode_when_explicitly_requested(self) -> None:
        with patch("claude_dj.agent.client.build_dj_mcp_server", return_value={"type": "sdk"}), patch.dict(
            os.environ,
            {"CLAUDE_DJ_CLAUDE_FAST_MODE": "1"},
            clear=True,
        ):
            options = build_agent_options(InMemoryTransitionStore(), narrator=object())

        self.assertEqual(options.extra_args, {"bare": None})

    async def test_agent_options_pass_reaction_source_to_mcp_server(self) -> None:
        reaction_source = object()
        store = InMemoryTransitionStore()
        narrator = object()

        with patch("claude_dj.agent.client.build_dj_mcp_server", return_value={"type": "sdk"}) as build_server:
            build_agent_options(store, narrator=narrator, reaction_source=reaction_source)

        build_server.assert_called_once_with(store, narrator, None, None, reaction_source)

    async def test_allowed_tools_include_seed_candidates(self) -> None:
        self.assertIn("mcp__dj__get_seed_candidates", build_allowed_tools())

    async def test_connects_before_start_query_and_disconnects_on_close(self) -> None:
        client = FakeSDKClient()
        agent = ClaudeDJ(client=client)

        await agent.connect()
        await agent.handle_start()
        await agent.disconnect()

        self.assertEqual(client.events, ["connect", "query", "receive", "disconnect"])

    async def test_connect_is_idempotent(self) -> None:
        client = FakeSDKClient()
        agent = ClaudeDJ(client=client)

        await agent.connect()
        await agent.connect()

        self.assertEqual(client.events, ["connect"])

    async def test_suppresses_claude_stream_logs_by_default(self) -> None:
        output = io.StringIO()
        client = FakeSDKClient(
            messages=[
                AssistantMessage(
                    content=[TextBlock(text="Starting the set.")],
                    model="fake-model",
                ),
            ]
        )
        agent = ClaudeDJ(client=client, output=output)

        await agent.connect()
        await agent.handle_start()

        self.assertEqual(output.getvalue(), "")

    async def test_logs_full_claude_stream_to_output_when_verbose(self) -> None:
        output = io.StringIO()
        client = FakeSDKClient(
            messages=[
                AssistantMessage(
                    content=[
                        TextBlock(text="Starting the set."),
                        ToolUseBlock(id="tool-1", name="mcp__dj__get_session_context", input={}),
                    ],
                    model="fake-model",
                ),
                UserMessage(
                    content=[
                        ToolResultBlock(
                            tool_use_id="tool-1",
                            content='{"seed_vibe":"playlist-informed autonomous start"}',
                        )
                    ],
                    tool_use_result={"seed_vibe": "playlist-informed autonomous start"},
                ),
                ResultMessage(
                    subtype="success",
                    duration_ms=10,
                    duration_api_ms=8,
                    is_error=False,
                    num_turns=1,
                    session_id="session-1",
                    stop_reason="end_turn",
                ),
            ]
        )
        agent = ClaudeDJ(client=client, output=output, verbose_claude=True)

        await agent.connect()
        await agent.handle_start()

        log = output.getvalue()
        self.assertIn("claude: assistant text Starting the set.", log)
        self.assertIn("claude: tool_use mcp__dj__get_session_context {}", log)
        self.assertIn("claude: tool_result", log)
        self.assertIn("playlist-informed autonomous start", log)
        self.assertIn("claude: result subtype=success is_error=False stop_reason=end_turn", log)

    async def test_logs_rate_limit_events_as_status_not_raw_error(self) -> None:
        output = io.StringIO()
        client = FakeSDKClient(
            messages=[
                RateLimitEvent(
                    rate_limit_info=RateLimitInfo(
                        status="allowed",
                        resets_at=1782024000,
                        rate_limit_type="five_hour",
                        overage_status="rejected",
                        overage_disabled_reason="out_of_credits",
                    ),
                    uuid="rate-limit-1",
                    session_id="session-1",
                )
            ]
        )
        agent = ClaudeDJ(client=client, output=output, verbose_claude=True)

        await agent.connect()
        await agent.handle_start()

        log = output.getvalue()
        self.assertIn(
            "claude: rate_limit status=allowed type=five_hour resets_at=1782024000 overage_status=rejected overage_disabled_reason=out_of_credits",
            log,
        )
        self.assertNotIn("claude: raw RateLimitEvent", log)

    async def test_raises_when_claude_turn_returns_error_result(self) -> None:
        client = FakeSDKClient(
            messages=[
                ResultMessage(
                    subtype="error_during_execution",
                    duration_ms=10,
                    duration_api_ms=8,
                    is_error=True,
                    num_turns=1,
                    session_id="session-1",
                    stop_reason="error",
                    errors=["missing api key"],
                    result="Claude Code failed before tool use.",
                ),
            ]
        )
        agent = ClaudeDJ(client=client)

        await agent.connect()

        with self.assertRaisesRegex(RuntimeError, "Claude turn failed"):
            await agent.handle_start()


if __name__ == "__main__":
    unittest.main()
