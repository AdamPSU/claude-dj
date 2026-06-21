import unittest
from unittest.mock import patch

from claude_dj.observability import observe_async, observe_run


class FakeSpan:
    def __init__(self, *, op: str, name: str) -> None:
        self.op = op
        self.name = name
        self.data: dict[str, object] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def set_data(self, key: str, value: object) -> None:
        self.data[key] = value


class ObservabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_context_adds_session_run_and_tool_sequence_to_tool_spans(self) -> None:
        transactions: list[FakeSpan] = []
        spans: list[FakeSpan] = []

        def fake_start_transaction(*, op: str, name: str):
            transaction = FakeSpan(op=op, name=name)
            transactions.append(transaction)
            return transaction

        def fake_start_span(*, op: str, name: str):
            span = FakeSpan(op=op, name=name)
            spans.append(span)
            return span

        async def noop() -> str:
            return "ok"

        async def run_tools() -> None:
            await observe_async(
                "claude_dj.mcp.get_session_context",
                op="mcp.tool",
                data={"tool": "get_session_context"},
                callback=noop,
            )
            await observe_async(
                "claude_dj.mcp.get_current_playback",
                op="mcp.tool",
                data={"tool": "get_current_playback"},
                callback=noop,
            )

        with patch("sentry_sdk.start_transaction", fake_start_transaction), patch(
            "sentry_sdk.start_span", fake_start_span
        ):
            await observe_run(
                "on_start",
                session_id="session-1",
                data={"hook": "on_start"},
                callback=run_tools,
            )

        self.assertEqual(transactions[0].op, "claude_dj.run")
        self.assertEqual(transactions[0].name, "claude_dj.run.on_start")
        self.assertEqual(transactions[0].data["claude_dj.session_id"], "session-1")
        self.assertEqual(transactions[0].data["claude_dj.run_type"], "on_start")
        self.assertEqual(transactions[0].data["tool_count"], 2)

        self.assertEqual([span.data["claude_dj.session_id"] for span in spans], ["session-1", "session-1"])
        self.assertEqual(spans[0].data["claude_dj.run_id"], spans[1].data["claude_dj.run_id"])
        self.assertEqual(spans[0].data["claude_dj.tool_index"], 1)
        self.assertEqual(spans[1].data["claude_dj.tool_index"], 2)
        self.assertEqual(spans[0].data["mcp.tool.name"], "get_session_context")
        self.assertEqual(spans[1].data["mcp.tool.name"], "get_current_playback")


if __name__ == "__main__":
    unittest.main()
