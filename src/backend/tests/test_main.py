import importlib
import io
import os
import sys
import unittest
from unittest.mock import patch

from claude_dj.reactions.monitor import ReactionEvent


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

    def test_main_defaults_demo_track_seconds_to_twenty(self) -> None:
        module = importlib.import_module("claude_dj.main")
        captured: dict[str, object] = {}

        async def fake_run_forever(**kwargs):
            captured.update(kwargs)

        with (
            patch.object(module, "load_dotenv", create=True),
            patch.object(module, "init_sentry"),
            patch.object(module, "run_forever", side_effect=fake_run_forever),
            patch.dict(os.environ, {}, clear=True),
        ):
            module.main(["--no-mascot"])

        self.assertEqual(captured["demo_track_seconds"], 20)


class MainBoundaryWatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_boundary_watcher_fires_once_when_current_track_reaches_zero(self) -> None:
        module = importlib.import_module("claude_dj.main")

        class FakePlayback:
            async def get_current_playback(self):
                return {
                    "current_track_id": "track-a",
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

    async def test_boundary_watcher_waits_for_pending_queue_when_cap_expires(self) -> None:
        module = importlib.import_module("claude_dj.main")

        class FakePlayback:
            def __init__(self) -> None:
                self.state = {
                    "current_track_id": "track-a",
                    "is_playing": True,
                    "progress_ms": 2_000,
                    "duration_ms": 2_000,
                    "seconds_remaining": 0,
                    "queue_track_ids": [],
                    "pending_queue_track_ids": [],
                }

            async def get_current_playback(self):
                return dict(self.state)

        class FakeRunner:
            def __init__(self) -> None:
                self.boundary_calls: list[str] = []

            async def on_track_boundary(self, *, ended_track_id: str) -> None:
                self.boundary_calls.append(ended_track_id)

        runner = FakeRunner()
        playback = FakePlayback()
        watcher = module.TrackBoundaryWatcher()

        before_queue = await watcher.maybe_handle_boundary(playback, runner)
        playback.state["pending_queue_track_ids"] = ["track-b"]
        after_queue = await watcher.maybe_handle_boundary(playback, runner)

        self.assertFalse(before_queue)
        self.assertTrue(after_queue)
        self.assertEqual(runner.boundary_calls, ["track-a"])


class MainHarnessTickTests(unittest.IsolatedAsyncioTestCase):
    async def test_tick_does_not_call_claude_when_no_reaction_event_occurs(self) -> None:
        module = importlib.import_module("claude_dj.main")

        class FakePlayback:
            async def get_current_playback(self):
                return {
                    "current_track_id": "track-a",
                    "seconds_remaining": 20,
                    "queue_track_ids": ["track-b"],
                    "pending_queue_track_ids": [],
                }

        class FakeRunner:
            def __init__(self) -> None:
                self.reaction_events: list[dict[str, object]] = []

            async def on_reaction_event(self, event: dict[str, object]) -> None:
                self.reaction_events.append(event)

        class FakeReactionSource:
            async def get_reaction_signal(self):
                return {"trend": "neutral", "confidence": 0.0}

        class FakeMonitor:
            async def poll(self, signal, playback):
                return None

        class FakeBoundaryWatcher:
            async def maybe_handle_boundary(self, playback, runner):
                return False

        runner = FakeRunner()
        output = io.StringIO()
        harness = module.DJHarness(runner, FakePlayback(), FakeReactionSource())

        handled = await module.run_harness_tick(harness, FakeBoundaryWatcher(), FakeMonitor(), output)

        self.assertEqual(handled, "idle")
        self.assertEqual(runner.reaction_events, [])
        self.assertIn(
            "ClaudeDJ queue state tick current=track-a queue=['track-b'] pending=[] seconds_remaining=20",
            output.getvalue(),
        )

    async def test_tick_calls_claude_once_for_sustained_negative_event(self) -> None:
        module = importlib.import_module("claude_dj.main")

        class FakePlayback:
            async def get_current_playback(self):
                return {"current_track_id": "track-a", "current_cluster": "rap", "seconds_remaining": 20}

        class FakeRunner:
            def __init__(self) -> None:
                self.reaction_events: list[dict[str, object]] = []

            async def on_reaction_event(self, event: dict[str, object]) -> None:
                self.reaction_events.append(event)

        class FakeReactionSource:
            async def get_reaction_signal(self):
                return {"trend": "negative", "confidence": 0.95, "score": -0.8}

        class FakeMonitor:
            async def poll(self, signal, playback):
                return ReactionEvent(
                    event_type="sustained_negative_reaction",
                    current_track_id="track-a",
                    current_cluster="rap",
                    duration_seconds=5.2,
                    signal=signal,
                )

        class FakeBoundaryWatcher:
            async def maybe_handle_boundary(self, playback, runner):
                return False

        runner = FakeRunner()
        output = io.StringIO()
        harness = module.DJHarness(runner, FakePlayback(), FakeReactionSource())

        handled = await module.run_harness_tick(harness, FakeBoundaryWatcher(), FakeMonitor(), output)

        self.assertEqual(handled, "reaction_event")
        self.assertEqual(len(runner.reaction_events), 1)
        self.assertEqual(runner.reaction_events[0]["event_type"], "sustained_negative_reaction")
        self.assertIn("ClaudeDJ reaction event planning starting", output.getvalue())

    async def test_tick_calls_claude_for_empty_queue_refresh(self) -> None:
        module = importlib.import_module("claude_dj.main")

        class FakePlayback:
            async def get_current_playback(self):
                return {
                    "current_track_id": "track-a",
                    "current_cluster": "rap",
                    "seconds_remaining": 25,
                    "queue_track_ids": [],
                    "pending_queue_track_ids": [],
                }

        class FakeRunner:
            def __init__(self) -> None:
                self.queue_refreshes: list[dict[str, object]] = []

            async def on_queue_refresh(self, playback: dict[str, object]) -> None:
                self.queue_refreshes.append(playback)

        class FakeReactionSource:
            async def get_reaction_signal(self):
                return {"trend": "neutral", "confidence": 0.0}

        class FakeMonitor:
            async def poll(self, signal, playback):
                return None

        class FakeBoundaryWatcher:
            async def maybe_handle_boundary(self, playback, runner):
                return False

        runner = FakeRunner()
        output = io.StringIO()
        harness = module.DJHarness(runner, FakePlayback(), FakeReactionSource())

        handled = await module.run_harness_tick(harness, FakeBoundaryWatcher(), FakeMonitor(), output)

        self.assertEqual(handled, "queue_refresh")
        self.assertEqual(len(runner.queue_refreshes), 1)
        self.assertIn("ClaudeDJ queue refresh planning starting", output.getvalue())

    async def test_tick_refreshes_queue_immediately_after_boundary_drains_queue(self) -> None:
        module = importlib.import_module("claude_dj.main")

        class FakePlayback:
            async def get_current_playback(self):
                return {
                    "current_track_id": "track-b",
                    "current_cluster": "rap",
                    "seconds_remaining": 2,
                    "queue_track_ids": [],
                    "pending_queue_track_ids": [],
                }

        class FakeRunner:
            def __init__(self) -> None:
                self.queue_refreshes: list[dict[str, object]] = []

            async def on_queue_refresh(self, playback: dict[str, object]) -> None:
                self.queue_refreshes.append(playback)

        class FakeReactionSource:
            async def get_reaction_signal(self):
                return {"trend": "neutral", "confidence": 0.0}

        class FakeReactionMonitor:
            async def poll(self, signal, playback):
                return None

        class FakeBoundaryWatcher:
            async def maybe_handle_boundary(self, playback, runner):
                return True

        runner = FakeRunner()
        output = io.StringIO()
        harness = module.DJHarness(runner, FakePlayback(), FakeReactionSource())

        handled = await module.run_harness_tick(
            harness,
            FakeBoundaryWatcher(),
            FakeReactionMonitor(),
            output,
            module.QueueRefreshMonitor(),
        )

        self.assertEqual(handled, "boundary_queue_refresh")
        self.assertEqual(len(runner.queue_refreshes), 1)
        self.assertIn("ClaudeDJ track boundary handled", output.getvalue())
        self.assertIn("ClaudeDJ queue refresh planning starting", output.getvalue())

    async def test_tick_runs_cluster_policy_after_boundary_before_queue_refresh(self) -> None:
        module = importlib.import_module("claude_dj.main")

        class FakePlayback:
            async def get_current_playback(self):
                return {
                    "current_track_id": "track-b",
                    "current_cluster": "rap",
                    "cluster_streak": 6,
                    "seconds_remaining": 2,
                    "queue_track_ids": [],
                    "pending_queue_track_ids": [],
                }

        class FakeRunner:
            def __init__(self) -> None:
                self.reaction_events: list[dict[str, object]] = []
                self.queue_refreshes: list[dict[str, object]] = []

            async def on_reaction_event(self, event: dict[str, object]) -> None:
                self.reaction_events.append(event)

            async def on_queue_refresh(self, playback: dict[str, object]) -> None:
                self.queue_refreshes.append(playback)

        class FakeReactionSource:
            async def get_reaction_signal(self):
                return {"trend": "neutral", "confidence": 0.0}

        class FakeReactionMonitor:
            async def poll(self, signal, playback):
                return None

        class FakeClusterPolicyMonitor:
            async def poll(self, playback):
                return ReactionEvent(
                    event_type="max_cluster_streak_reached",
                    current_track_id="track-b",
                    current_cluster="rap",
                    duration_seconds=0.0,
                    signal={"trend": "neutral", "source": "cluster_policy"},
                )

        class FakeBoundaryWatcher:
            async def maybe_handle_boundary(self, playback, runner):
                return True

        runner = FakeRunner()
        output = io.StringIO()
        harness = module.DJHarness(runner, FakePlayback(), FakeReactionSource())

        handled = await module.run_harness_tick(
            harness,
            FakeBoundaryWatcher(),
            FakeReactionMonitor(),
            output,
            module.QueueRefreshMonitor(),
            cluster_policy_monitor=FakeClusterPolicyMonitor(),
        )

        self.assertEqual(handled, "boundary_reaction_event")
        self.assertEqual(runner.reaction_events[0]["event_type"], "max_cluster_streak_reached")
        self.assertEqual(runner.queue_refreshes, [])
        self.assertIn("ClaudeDJ reaction event planning starting", output.getvalue())

    async def test_tick_calls_claude_for_cluster_policy_event_before_queue_refresh(self) -> None:
        module = importlib.import_module("claude_dj.main")

        class FakePlayback:
            async def get_current_playback(self):
                return {
                    "current_track_id": "track-a",
                    "current_cluster": "rap",
                    "cluster_streak": 6,
                    "seconds_remaining": 25,
                    "queue_track_ids": [],
                    "pending_queue_track_ids": [],
                }

        class FakeRunner:
            def __init__(self) -> None:
                self.reaction_events: list[dict[str, object]] = []
                self.queue_refreshes: list[dict[str, object]] = []

            async def on_reaction_event(self, event: dict[str, object]) -> None:
                self.reaction_events.append(event)

            async def on_queue_refresh(self, playback: dict[str, object]) -> None:
                self.queue_refreshes.append(playback)

        class FakeReactionSource:
            async def get_reaction_signal(self):
                return {"trend": "neutral", "confidence": 0.0}

        class FakeReactionMonitor:
            async def poll(self, signal, playback):
                return None

        class FakeClusterPolicyMonitor:
            async def poll(self, playback):
                return ReactionEvent(
                    event_type="max_cluster_streak_reached",
                    current_track_id="track-a",
                    current_cluster="rap",
                    duration_seconds=0.0,
                    signal={"trend": "neutral", "source": "cluster_policy"},
                )

        class FakeBoundaryWatcher:
            async def maybe_handle_boundary(self, playback, runner):
                return False

        runner = FakeRunner()
        output = io.StringIO()
        harness = module.DJHarness(runner, FakePlayback(), FakeReactionSource())

        handled = await module.run_harness_tick(
            harness,
            FakeBoundaryWatcher(),
            FakeReactionMonitor(),
            output,
            cluster_policy_monitor=FakeClusterPolicyMonitor(),
        )

        self.assertEqual(handled, "reaction_event")
        self.assertEqual(runner.reaction_events[0]["event_type"], "max_cluster_streak_reached")
        self.assertEqual(runner.queue_refreshes, [])

    async def test_empty_queue_refresh_fires_once_per_track(self) -> None:
        module = importlib.import_module("claude_dj.main")
        monitor = module.QueueRefreshMonitor()
        playback = {
            "current_track_id": "track-a",
            "queue_track_ids": [],
            "pending_queue_track_ids": [],
        }

        first = monitor.should_refresh(playback)
        second = monitor.should_refresh(playback)
        third = monitor.should_refresh({**playback, "current_track_id": "track-b"})

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertTrue(third)


class MainReactionRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_reaction_runtime_defaults_to_webcam_reactor(self) -> None:
        module = importlib.import_module("claude_dj.main")

        with patch.dict(os.environ, {}, clear=True):
            runtime = module.build_reaction_runtime()

        signal = await runtime.source.get_reaction_signal()

        self.assertIsNotNone(runtime.reactor)
        self.assertIsNotNone(runtime.reactor.frame_source)
        self.assertIs(runtime.preview_worker, runtime.reactor.frame_source)
        self.assertTrue(runtime.reactor.frame_source.show_preview)
        self.assertFalse(signal["stub"])

    async def test_reaction_runtime_can_opt_out_of_webcam_capture(self) -> None:
        module = importlib.import_module("claude_dj.main")

        with patch.dict(
            os.environ,
            {"CLAUDE_DJ_NO_WEBCAM": "1"},
            clear=True,
        ):
            runtime = module.build_reaction_runtime()

        signal = await runtime.source.get_reaction_signal()

        self.assertIsNotNone(runtime.reactor)
        self.assertIsNone(runtime.reactor.frame_source)
        self.assertIsNone(runtime.preview_worker)
        self.assertFalse(signal["stub"])

    async def test_cluster_policy_monitor_defaults_to_two_song_demo_run(self) -> None:
        module = importlib.import_module("claude_dj.main")

        with patch.dict(os.environ, {}, clear=True):
            monitor = module.build_cluster_policy_monitor()

        self.assertEqual(monitor.max_cluster_run, 2)

    async def test_cluster_policy_monitor_allows_env_override(self) -> None:
        module = importlib.import_module("claude_dj.main")

        with patch.dict(os.environ, {"CLAUDE_DJ_MAX_CLUSTER_RUN": "6"}, clear=True):
            monitor = module.build_cluster_policy_monitor()

        self.assertEqual(monitor.max_cluster_run, 6)


if __name__ == "__main__":
    unittest.main()
