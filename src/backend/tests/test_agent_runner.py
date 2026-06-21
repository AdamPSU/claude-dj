import unittest
from unittest.mock import patch

from claude_dj.agent.runner import DJAgentRunner


class FakeAgent:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def connect(self) -> None:
        self.events.append("connect")

    async def disconnect(self) -> None:
        self.events.append("disconnect")

    async def handle_start(self) -> None:
        self.events.append("start")

    async def handle_reaction_event(self, event: dict[str, object]) -> None:
        self.events.append(f"reaction:{event['event_type']}")

    async def handle_queue_refresh(self, playback: dict[str, object]) -> None:
        self.events.append(f"queue:{playback['current_track_id']}")


class FakeBoundary:
    async def on_track_boundary(self, ended_track_id: str) -> None:
        return None


class RunnerObservabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_hooks_run_inside_stable_session_runs(self) -> None:
        calls: list[dict[str, object]] = []

        async def fake_observe_run(run_type, *, session_id, data, callback):
            calls.append({"run_type": run_type, "session_id": session_id, "data": data})
            await callback()

        agent = FakeAgent()
        runner = DJAgentRunner(agent, FakeBoundary())

        with patch("claude_dj.agent.runner.observe_run", fake_observe_run):
            await runner.on_start()
            await runner.on_reaction_event({"event_type": "sustained_negative_reaction"})
            await runner.on_queue_refresh({"current_track_id": "track-a"})

        self.assertEqual(agent.events, ["start", "reaction:sustained_negative_reaction", "queue:track-a"])
        self.assertEqual(
            [call["run_type"] for call in calls],
            ["on_start", "on_reaction_event", "on_queue_refresh"],
        )
        self.assertTrue(calls[0]["session_id"])
        self.assertEqual(calls[0]["session_id"], calls[1]["session_id"])
        self.assertEqual(calls[1]["data"], {"hook": "on_reaction_event", "event_type": "sustained_negative_reaction"})
        self.assertEqual(calls[2]["session_id"], calls[0]["session_id"])
        self.assertEqual(calls[2]["data"], {"hook": "on_queue_refresh", "current_track_id": "track-a"})


if __name__ == "__main__":
    unittest.main()
