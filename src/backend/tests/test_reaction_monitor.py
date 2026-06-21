import unittest

from claude_dj.reactions.monitor import ClusterPolicyMonitor, ReactionMonitor


class SequenceClock:
    def __init__(self, times: list[float]) -> None:
        self.times = list(times)

    def __call__(self) -> float:
        if not self.times:
            raise AssertionError("clock exhausted")
        return self.times.pop(0)


class ReactionMonitorTests(unittest.IsolatedAsyncioTestCase):
    async def test_sustained_negative_signal_triggers_once_after_threshold(self) -> None:
        clock = SequenceClock([0.0, 3.0, 5.1, 6.0])
        monitor = ReactionMonitor(negative_seconds=5.0, cooldown_seconds=60.0, clock=clock)
        playback = {
            "current_track_id": "track-a",
            "current_cluster": "rap",
            "queue_track_ids": ["track-b"],
            "pending_queue_track_ids": [],
        }
        signal = {"trend": "negative", "confidence": 0.9, "score": -0.8, "source": "fake_camera"}

        self.assertIsNone(await monitor.poll(signal, playback))
        self.assertIsNone(await monitor.poll(signal, playback))
        event = await monitor.poll(signal, playback)
        duplicate = await monitor.poll(signal, playback)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.event_type, "sustained_negative_reaction")
        self.assertEqual(event.current_track_id, "track-a")
        self.assertEqual(event.current_cluster, "rap")
        self.assertGreaterEqual(event.duration_seconds, 5.0)
        self.assertIsNone(duplicate)

    async def test_negative_timer_resets_when_signal_recovers(self) -> None:
        clock = SequenceClock([0.0, 4.0, 5.1, 10.2])
        monitor = ReactionMonitor(negative_seconds=5.0, cooldown_seconds=60.0, clock=clock)
        playback = {"current_track_id": "track-a", "pending_queue_track_ids": []}

        self.assertIsNone(await monitor.poll({"trend": "negative", "confidence": 0.9}, playback))
        self.assertIsNone(await monitor.poll({"trend": "neutral", "confidence": 0.9}, playback))
        self.assertIsNone(await monitor.poll({"trend": "negative", "confidence": 0.9}, playback))
        event = await monitor.poll({"trend": "negative", "confidence": 0.9}, playback)

        self.assertGreaterEqual(event.duration_seconds, 5.0)

    async def test_pending_transition_suppresses_negative_event(self) -> None:
        clock = SequenceClock([0.0, 6.0])
        monitor = ReactionMonitor(negative_seconds=5.0, cooldown_seconds=60.0, clock=clock)
        playback = {
            "current_track_id": "track-a",
            "pending_queue_track_ids": ["track-c"],
        }
        signal = {"trend": "negative", "confidence": 0.9}

        self.assertIsNone(await monitor.poll(signal, playback))
        self.assertIsNone(await monitor.poll(signal, playback))

    async def test_late_negative_signal_defers_until_following_song(self) -> None:
        clock = SequenceClock([0.0, 5.1, 6.0])
        monitor = ReactionMonitor(negative_seconds=5.0, cooldown_seconds=60.0, clock=clock)
        signal = {"trend": "negative", "confidence": 0.9, "score": -0.8}
        late_playback = {
            "current_track_id": "track-a",
            "current_cluster": "rap",
            "progress_ms": 75_000,
            "duration_ms": 100_000,
            "queue_track_ids": ["track-b"],
            "pending_queue_track_ids": [],
        }
        next_playback = {
            "current_track_id": "track-b",
            "current_cluster": "rap",
            "progress_ms": 1_000,
            "duration_ms": 100_000,
            "queue_track_ids": ["track-c"],
            "pending_queue_track_ids": [],
        }

        self.assertIsNone(await monitor.poll(signal, late_playback))
        self.assertIsNone(await monitor.poll(signal, late_playback))
        event = await monitor.poll({"trend": "neutral", "confidence": 0.0}, next_playback)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.current_track_id, "track-b")
        self.assertEqual(event.metadata["deferred_from_track_id"], "track-a")


class ClusterPolicyMonitorTests(unittest.IsolatedAsyncioTestCase):
    async def test_random_cluster_target_triggers_at_selected_run_length(self) -> None:
        monitor = ClusterPolicyMonitor(min_cluster_run=2, max_cluster_run=4, choose_target=lambda low, high: 3)
        base_playback = {
            "current_track_id": "track-a",
            "current_cluster": "rap",
            "progress_ms": 20_000,
            "duration_ms": 100_000,
            "queue_track_ids": ["track-b"],
            "pending_queue_track_ids": [],
        }

        self.assertIsNone(await monitor.poll({**base_playback, "cluster_streak": 2}))
        event = await monitor.poll({**base_playback, "cluster_streak": 3})

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.metadata["target_cluster_run"], 3)
        self.assertEqual(event.metadata["min_cluster_run"], 2)
        self.assertEqual(event.metadata["max_cluster_run"], 4)

    async def test_random_cluster_target_resets_for_new_cluster(self) -> None:
        chosen_targets = [2, 4]

        def choose_target(low: int, high: int) -> int:
            return chosen_targets.pop(0)

        monitor = ClusterPolicyMonitor(min_cluster_run=2, max_cluster_run=4, choose_target=choose_target)

        first_event = await monitor.poll(
            {
                "current_track_id": "track-a",
                "current_cluster": "rap",
                "cluster_streak": 2,
                "progress_ms": 20_000,
                "duration_ms": 100_000,
                "queue_track_ids": ["track-b"],
                "pending_queue_track_ids": [],
            }
        )
        second_event = await monitor.poll(
            {
                "current_track_id": "track-c",
                "current_cluster": "house",
                "cluster_streak": 3,
                "progress_ms": 20_000,
                "duration_ms": 100_000,
                "queue_track_ids": ["track-d"],
                "pending_queue_track_ids": [],
            }
        )

        self.assertIsNotNone(first_event)
        self.assertIsNone(second_event)
        self.assertEqual(chosen_targets, [])

    async def test_max_cluster_streak_triggers_shift_event(self) -> None:
        monitor = ClusterPolicyMonitor(max_cluster_run=6)
        playback = {
            "current_track_id": "track-a",
            "current_cluster": "rap",
            "cluster_streak": 6,
            "progress_ms": 20_000,
            "duration_ms": 100_000,
            "queue_track_ids": ["track-b"],
            "pending_queue_track_ids": [],
        }

        event = await monitor.poll(playback)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.event_type, "max_cluster_streak_reached")
        self.assertEqual(event.current_cluster, "rap")
        self.assertEqual(event.metadata["cluster_streak"], 6)

    async def test_late_max_cluster_streak_defers_until_following_song(self) -> None:
        monitor = ClusterPolicyMonitor(max_cluster_run=6)
        late_playback = {
            "current_track_id": "track-a",
            "current_cluster": "rap",
            "cluster_streak": 6,
            "progress_ms": 75_000,
            "duration_ms": 100_000,
            "queue_track_ids": ["track-b"],
            "pending_queue_track_ids": [],
        }
        next_playback = {
            "current_track_id": "track-b",
            "current_cluster": "rap",
            "cluster_streak": 7,
            "progress_ms": 1_000,
            "duration_ms": 100_000,
            "queue_track_ids": ["track-c"],
            "pending_queue_track_ids": [],
        }

        self.assertIsNone(await monitor.poll(late_playback))
        event = await monitor.poll(next_playback)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.current_track_id, "track-b")
        self.assertEqual(event.metadata["deferred_from_track_id"], "track-a")


if __name__ == "__main__":
    unittest.main()
