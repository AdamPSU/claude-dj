import importlib
import os
import sys
import unittest
from unittest.mock import patch


class MainCLITests(unittest.TestCase):
    def test_cli_module_import_does_not_initialize_sentry(self) -> None:
        sys.modules.pop("claude_dj.main", None)
        from claude_dj import observability

        with (
            patch.object(observability, "_initialized", False),
            patch("sentry_sdk.init") as init_mock,
            patch.dict(os.environ, {"SENTRY_DSN": "https://public@example.com/1"}),
        ):
            importlib.import_module("claude_dj.main")

        init_mock.assert_not_called()

    def test_env_flag_accepts_common_truthy_values(self) -> None:
        from claude_dj.main import env_flag

        for value in ["1", "true", "TRUE", "yes", "on"]:
            with self.subTest(value=value), patch.dict(os.environ, {"FLAG": value}):
                self.assertTrue(env_flag("FLAG"))

    def test_env_flag_defaults_false(self) -> None:
        from claude_dj.main import env_flag

        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(env_flag("FLAG"))

    def test_main_initializes_sentry_when_running_cli(self) -> None:
        module = importlib.import_module("claude_dj.main")

        def stop_after_creating_coroutine(coroutine):
            coroutine.close()
            raise KeyboardInterrupt

        with patch.object(module, "init_sentry") as init_mock, patch.object(
            module.asyncio,
            "run",
            side_effect=stop_after_creating_coroutine,
        ):
            module.main([])

        init_mock.assert_called_once_with()

    def test_main_loads_dotenv_before_initializing_sentry(self) -> None:
        module = importlib.import_module("claude_dj.main")
        events: list[str] = []

        def stop_after_creating_coroutine(coroutine):
            coroutine.close()
            raise KeyboardInterrupt

        with (
            patch.object(module, "load_dotenv", create=True, side_effect=lambda path: events.append(f"dotenv:{path}")),
            patch.object(module, "init_sentry", side_effect=lambda: events.append("sentry")),
            patch.object(
                module.asyncio,
                "run",
                side_effect=stop_after_creating_coroutine,
            ),
        ):
            module.main([])

        self.assertEqual(events, ["dotenv:.env", "sentry"])

    def test_main_passes_demo_track_seconds_to_harness(self) -> None:
        module = importlib.import_module("claude_dj.main")
        captured: dict[str, object] = {}

        async def fake_run_forever(**kwargs):
            captured.update(kwargs)

        with (
            patch.object(module, "load_dotenv", create=True),
            patch.object(module, "init_sentry"),
            patch.object(module, "run_forever", side_effect=fake_run_forever),
        ):
            module.main(["--demo-track-seconds", "30", "--no-mascot"])

        self.assertEqual(captured["demo_track_seconds"], 30)
        self.assertFalse(captured["launch_mascot"])


class MainBoundaryWatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_boundary_watcher_fires_once_when_current_track_reaches_zero(self) -> None:
        module = importlib.import_module("claude_dj.main")

        class FakePlayback:
            async def get_current_playback(self):
                return {
                    "current_track_id": "track-a",
                    "seconds_remaining": 0,
                }

        class FakeRunner:
            def __init__(self) -> None:
                self.boundary_calls: list[str] = []

            async def on_track_boundary(self, *, ended_track_id: str) -> None:
                self.boundary_calls.append(ended_track_id)

        runner = FakeRunner()
        watcher = module.TrackBoundaryWatcher()

        first = await watcher.maybe_handle_boundary(FakePlayback(), runner)
        second = await watcher.maybe_handle_boundary(FakePlayback(), runner)

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(runner.boundary_calls, ["track-a"])

    async def test_boundary_watcher_handles_spotify_reset_after_missing_zero_window(self) -> None:
        module = importlib.import_module("claude_dj.main")

        class FakePlayback:
            def __init__(self) -> None:
                self.states = [
                    {
                        "current_track_id": "track-a",
                        "is_playing": True,
                        "progress_ms": 219_000,
                        "duration_ms": 220_000,
                        "seconds_remaining": 1,
                        "queue_track_ids": ["track-b"],
                        "pending_queue_track_ids": [],
                    },
                    {
                        "current_track_id": "track-a",
                        "is_playing": False,
                        "progress_ms": 0,
                        "duration_ms": 220_000,
                        "seconds_remaining": 220,
                        "queue_track_ids": ["track-b"],
                        "pending_queue_track_ids": [],
                    },
                ]

            async def get_current_playback(self):
                return self.states.pop(0)

        class FakeRunner:
            def __init__(self) -> None:
                self.boundary_calls: list[str] = []

            async def on_track_boundary(self, *, ended_track_id: str) -> None:
                self.boundary_calls.append(ended_track_id)

        runner = FakeRunner()
        watcher = module.TrackBoundaryWatcher()
        playback = FakePlayback()

        before_end = await watcher.maybe_handle_boundary(playback, runner)
        after_spotify_reset = await watcher.maybe_handle_boundary(playback, runner)

        self.assertFalse(before_end)
        self.assertTrue(after_spotify_reset)
        self.assertEqual(runner.boundary_calls, ["track-a"])

    async def test_boundary_watcher_fires_when_demo_track_cap_expires(self) -> None:
        module = importlib.import_module("claude_dj.main")

        class FakePlayback:
            async def get_current_playback(self):
                return {
                    "current_track_id": "track-a",
                    "is_playing": True,
                    "progress_ms": 30_000,
                    "duration_ms": 30_000,
                    "seconds_remaining": 0,
                    "queue_track_ids": ["track-b"],
                    "pending_queue_track_ids": [],
                }

        class FakeRunner:
            def __init__(self) -> None:
                self.boundary_calls: list[str] = []

            async def on_track_boundary(self, *, ended_track_id: str) -> None:
                self.boundary_calls.append(ended_track_id)

        runner = FakeRunner()
        handled = await module.TrackBoundaryWatcher().maybe_handle_boundary(FakePlayback(), runner)

        self.assertTrue(handled)
        self.assertEqual(runner.boundary_calls, ["track-a"])


if __name__ == "__main__":
    unittest.main()
