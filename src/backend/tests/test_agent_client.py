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
from claude_dj.agent.prompts import DJ_SYSTEM_PROMPT
from claude_dj.transition import InMemoryTransitionStore


class FakeSDKClient:
    def __init__(self, messages: list[object] | None = None) -> None:
        self.connected = False
        self.events: list[str] = []
        self.prompts: list[str] = []
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
        self.prompts.append(prompt)

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

    async def test_start_prompt_prefers_session_initial_seed_before_seed_candidates(self) -> None:
        client = FakeSDKClient()
        agent = ClaudeDJ(client=client)

        await agent.connect()
        await agent.handle_start()

        prompt = client.prompts[-1]
        self.assertIn("initial_seed_track_id", prompt)
        self.assertIn("get_seed_candidates only if", prompt)
        self.assertIn("Call search_track_embeddings", prompt)
        self.assertIn("Choose 2-4 tracks", prompt)
        self.assertIn('reason="startup_set"', prompt)
        self.assertLess(prompt.index("Call narrate"), prompt.index("Call play_track"))

    async def test_system_prompt_sets_concise_dj_personality_guardrails(self) -> None:
        self.assertIn("personal music guide", DJ_SYSTEM_PROMPT)
        self.assertIn("human, paced, and conversational", DJ_SYSTEM_PROMPT)
        self.assertIn("nostalgia, discovery, mood, energy", DJ_SYSTEM_PROMPT)
        self.assertIn("Do not invent artist facts", DJ_SYSTEM_PROMPT)
        self.assertIn("Do not mention internal tools", DJ_SYSTEM_PROMPT)
        self.assertIn("Blend a familiar anchor", DJ_SYSTEM_PROMPT)
        self.assertIn("what is changing musically", DJ_SYSTEM_PROMPT)

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

    async def test_reaction_event_prompt_instructs_negative_shift_without_playing_track(self) -> None:
        client = FakeSDKClient()
        agent = ClaudeDJ(client=client)

        await agent.connect()
        await agent.handle_reaction_event(
            {
                "event_type": "sustained_negative_reaction",
                "current_track_id": "deezer:123",
                "current_cluster": "rap_hip_hop",
                "duration_seconds": 5.2,
                "signal": {"trend": "negative", "confidence": 0.95, "score": -0.8},
            }
        )

        prompt = client.prompts[-1]
        self.assertIn("sustained_negative_reaction", prompt)
        self.assertIn("deezer:123", prompt)
        self.assertIn("rap_hip_hop", prompt)
        self.assertIn('signal="negative"', prompt)
        self.assertIn('mode="shift"', prompt)
        self.assertIn('timing="after_current_track"', prompt)
        self.assertIn("mark_track_feedback", prompt)
        self.assertIn("reason=\"reaction_shift\"", prompt)
        self.assertIn("Do not call play_track", prompt)

    async def test_cluster_policy_prompt_shifts_without_negative_feedback_signal(self) -> None:
        client = FakeSDKClient()
        agent = ClaudeDJ(client=client)

        await agent.connect()
        await agent.handle_reaction_event(
            {
                "event_type": "max_cluster_streak_reached",
                "current_track_id": "deezer:123",
                "current_cluster": "rap_hip_hop",
                "duration_seconds": 0.0,
                "signal": {"trend": "neutral", "source": "cluster_policy"},
            }
        )

        prompt = client.prompts[-1]
        self.assertIn("max_cluster_streak_reached", prompt)
        self.assertIn('signal="neutral"', prompt)
        self.assertIn('mode="shift"', prompt)
        self.assertIn('timing="after_current_track"', prompt)
        self.assertIn("freshening the set", prompt)

    async def test_queue_refresh_prompt_refills_without_forcing_narration(self) -> None:
        client = FakeSDKClient()
        agent = ClaudeDJ(client=client)

        await agent.connect()
        await agent.handle_queue_refresh(
            {
                "current_track_id": "deezer:123",
                "current_cluster": "rap_hip_hop",
                "queue_track_ids": [],
                "pending_queue_track_ids": [],
                "seconds_remaining": 25,
            }
        )

        prompt = client.prompts[-1]
        self.assertIn("queue_refresh", prompt)
        self.assertIn("deezer:123", prompt)
        self.assertIn('mode="similar"', prompt)
        self.assertIn('timing="after_current_track"', prompt)
        self.assertIn("Only call narrate", prompt)
        self.assertIn("reason=\"same_lane_refill\"", prompt)
        self.assertIn("Do not call play_track", prompt)


if __name__ == "__main__":
    unittest.main()
