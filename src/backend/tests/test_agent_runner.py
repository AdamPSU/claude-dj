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

    async def handle_mid_song_prepare(self, progress_percent: int) -> None:
        self.events.append(f"mid:{progress_percent}")


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
            await runner.on_mid_song_prepare(progress_percent=55)

        self.assertEqual(agent.events, ["start", "mid:55"])
        self.assertEqual([call["run_type"] for call in calls], ["on_start", "on_mid_song_prepare"])
        self.assertTrue(calls[0]["session_id"])
        self.assertEqual(calls[0]["session_id"], calls[1]["session_id"])
        self.assertEqual(calls[1]["data"], {"hook": "on_mid_song_prepare", "progress_percent": 55})


if __name__ == "__main__":
    unittest.main()
