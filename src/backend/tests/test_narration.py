import asyncio
import signal
import subprocess
import unittest
from unittest.mock import patch

from claude_dj.mcp.handlers import DJToolHandlers
from claude_dj.mcp.narration import (
    DeepgramResponse,
    DeepgramNarrator,
    EphemeralNarrationStore,
    LocalNarrationPlayer,
    NarrationAudio,
    UrlLibDeepgramRequester,
)
from claude_dj.transition import BoundaryExecutor, InMemoryTransitionStore, TransitionPlan


class FakeRequester:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def post(self, url: str, *, headers: dict[str, str], json: dict[str, str]) -> DeepgramResponse:
        self.calls.append({"url": url, "headers": headers, "json": json})
        return DeepgramResponse(audio=b"wav-bytes", content_type="audio/wav")


class FakeNarrator:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def generate(self, text: str) -> NarrationAudio:
        self.texts.append(text)
        return NarrationAudio(
            id=f"narration-{len(self.texts)}",
            text=text,
            audio=b"fake-audio",
            content_type="audio/wav",
            model="fake-model",
        )


class FakeNarrationPlayer:
    def __init__(self) -> None:
        self.played: list[NarrationAudio] = []

    def play(self, narration: NarrationAudio) -> None:
        self.played.append(narration)


class FakeBoundaryAdapter:
    def __init__(self) -> None:
        self.volume = 100
        self.volume_events: list[int] = []
        self.played_tracks: list[str] = []
        self.played_narrations: list[str] = []
        self.next_queued_track_id: str | None = None
        self.events: list[str] = []

    async def get_music_volume(self) -> int:
        return self.volume

    async def set_music_volume(self, volume_percent: int) -> None:
        self.volume = volume_percent
        self.volume_events.append(volume_percent)
        self.events.append(f"volume:{volume_percent}")

    async def pause_music(self) -> None:
        self.events.append("pause")

    async def resume_music(self) -> None:
        self.events.append("resume")

    async def play_track(self, track_id: str) -> None:
        self.played_tracks.append(track_id)
        self.events.append(f"track:{track_id}")

    async def play_next_queued_track(self) -> str | None:
        if self.next_queued_track_id is None:
            return None
        self.played_tracks.append(self.next_queued_track_id)
        return self.next_queued_track_id

    async def play_narration(self, narration_id: str) -> None:
        self.played_narrations.append(narration_id)
        self.events.append(f"narration:{narration_id}")


class RecordingSleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)


class NarrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_narration_player_ignores_interrupted_afplay(self) -> None:
        player = LocalNarrationPlayer()
        narration = NarrationAudio(
            id="narration-1",
            text="Interrupted playback.",
            audio=b"fake-mp3",
            content_type="audio/mpeg",
            model="fake-model",
        )

        def interrupted_afplay(*args, **kwargs):
            raise subprocess.CalledProcessError(-signal.SIGINT, args[0])

        with patch("claude_dj.mcp.narration.platform.system", return_value="Darwin"), patch(
            "claude_dj.mcp.narration.subprocess.run",
            interrupted_afplay,
        ):
            player.play(narration)

    async def test_url_lib_deepgram_requester_uses_timeout(self) -> None:
        timeouts = []

        class FakeUrlOpenResponse:
            headers = {"Content-Type": "audio/mpeg"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback) -> None:
                return None

            def read(self) -> bytes:
                return b"audio"

        def fake_urlopen(request, timeout=None):
            timeouts.append(timeout)
            return FakeUrlOpenResponse()

        requester = UrlLibDeepgramRequester(request_timeout_seconds=7.0)

        with patch("claude_dj.mcp.narration.urlopen", fake_urlopen):
            response = await requester.post(
                "https://api.deepgram.com/v1/speak?model=test",
                headers={"Authorization": "Token test"},
                json={"text": "hello"},
            )

        self.assertEqual(response.audio, b"audio")
        self.assertEqual(timeouts, [7.0])

    async def test_deepgram_narrator_posts_text_and_stores_ephemeral_audio(self) -> None:
        store = EphemeralNarrationStore()
        requester = FakeRequester()
        narrator = DeepgramNarrator(
            api_key="dg-test-key",
            model="aura-2-luna-en",
            speed=1.3,
            store=store,
            requester=requester,
        )

        audio = await narrator.generate("This groove is working, so I am moving one step over.")

        self.assertEqual(audio.id, "narration-1")
        self.assertEqual(audio.audio, b"wav-bytes")
        self.assertEqual(audio.content_type, "audio/wav")
        self.assertEqual(audio.model, "aura-2-luna-en")
        self.assertIs(store.get("narration-1"), audio)
        self.assertEqual(
            requester.calls,
            [
                {
                    "url": "https://api.deepgram.com/v1/speak?model=aura-2-luna-en&speed=1.3",
                    "headers": {
                        "Authorization": "Token dg-test-key",
                        "Content-Type": "application/json",
                    },
                    "json": {"text": "This groove is working, so I am moving one step over."},
                }
            ],
        )

        store.delete("narration-1")
        self.assertIsNone(store.get("narration-1"))

    async def test_narrate_immediate_generates_audio_without_transition_plan(self) -> None:
        store = InMemoryTransitionStore()
        narrator = FakeNarrator()
        player = FakeNarrationPlayer()
        handlers = DJToolHandlers(store, narrator, narration_player=player)

        result = await handlers.narrate(
            text="I found a warm start. Let me open with this pocket.",
            mode="immediate",
            reason="startup",
        )

        self.assertEqual(result["audio_id"], "narration-1")
        self.assertTrue(result["displayed"])
        self.assertTrue(result["spoken"])
        self.assertTrue(result["played"])
        self.assertEqual(player.played[0].id, "narration-1")
        self.assertIsNone(store.get_ready_plan("stub-current-track"))

    async def test_narrate_prepare_records_ready_transition_plan(self) -> None:
        store = InMemoryTransitionStore()
        narrator = FakeNarrator()
        handlers = DJToolHandlers(store, narrator)

        result = await handlers.narrate(
            text="Ouch, that did not land. Let me put you onto something smoother.",
            mode="prepare",
            reason="genre_shift_bridge",
            timing="after_current_track",
            current_track_id="track-a",
            next_track_id="track-b",
            track_ids=["track-b", "track-c", "track-d"],
        )

        plan = store.get_ready_plan("track-a")
        self.assertEqual(result["audio_id"], "narration-1")
        self.assertTrue(result["ready"])
        self.assertTrue(result["spoken"])
        self.assertEqual(narrator.texts, ["Ouch, that did not land. Let me put you onto something smoother."])
        self.assertIsNotNone(plan)
        self.assertEqual(plan.next_track_id, "track-b")
        self.assertEqual(plan.track_ids, ["track-b", "track-c", "track-d"])

    async def test_narrate_prepare_validates_transition_fields_before_tts(self) -> None:
        narrator = FakeNarrator()
        handlers = DJToolHandlers(InMemoryTransitionStore(), narrator)

        with self.assertRaises(ValueError):
            await handlers.narrate(
                text="Let me switch this up.",
                mode="prepare",
                reason="genre_shift_bridge",
            )

        self.assertEqual(narrator.texts, [])

    async def test_boundary_executes_ready_transition_without_deepgram(self) -> None:
        store = InMemoryTransitionStore()
        adapter = FakeBoundaryAdapter()
        sleeper = RecordingSleeper()
        store.save(
            TransitionPlan(
                current_track_id="track-a",
                next_track_id="track-b",
                track_ids=["track-b", "track-c", "track-d"],
                narration_id="narration-1",
            )
        )

        await BoundaryExecutor(store, adapter, sleep=sleeper.sleep).on_track_boundary(ended_track_id="track-a")

        self.assertEqual(
            adapter.volume_events,
            [90, 80, 70, 60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        )
        self.assertEqual(adapter.played_tracks, ["track-b"])
        self.assertEqual(adapter.played_narrations, ["narration-1"])
        self.assertEqual(
            adapter.events,
            [
                "volume:90",
                "volume:80",
                "volume:70",
                "volume:60",
                "volume:50",
                "volume:40",
                "volume:30",
                "volume:20",
                "volume:10",
                "volume:0",
                "track:track-b",
                "pause",
                "narration:narration-1",
                "resume",
                "volume:10",
                "volume:20",
                "volume:30",
                "volume:40",
                "volume:50",
                "volume:60",
                "volume:70",
                "volume:80",
                "volume:90",
                "volume:100",
            ],
        )
        self.assertEqual(sleeper.delays, [0.1] * 20)
        self.assertIsNone(store.get_ready_plan("track-a"))

    async def test_boundary_falls_back_to_next_queued_track_without_ready_transition(self) -> None:
        store = InMemoryTransitionStore()
        adapter = FakeBoundaryAdapter()
        adapter.next_queued_track_id = "track-b"
        sleeper = RecordingSleeper()

        await BoundaryExecutor(store, adapter, sleep=sleeper.sleep).on_track_boundary(ended_track_id="track-a")

        self.assertEqual(adapter.played_tracks, ["track-b"])
        self.assertEqual(adapter.played_narrations, [])
        self.assertEqual(
            adapter.volume_events,
            [90, 80, 70, 60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        )
        self.assertEqual(sleeper.delays, [0.1] * 20)

    async def test_boundary_restores_volume_when_cancelled_during_fade_out(self) -> None:
        store = InMemoryTransitionStore()
        adapter = FakeBoundaryAdapter()
        adapter.next_queued_track_id = "track-b"

        class CancellingSleeper:
            def __init__(self) -> None:
                self.calls = 0

            async def sleep(self, delay: float) -> None:
                self.calls += 1
                if self.calls == 2:
                    raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await BoundaryExecutor(store, adapter, sleep=CancellingSleeper().sleep).on_track_boundary(
                ended_track_id="track-a"
            )

        self.assertEqual(adapter.volume_events, [90, 80, 100])
        self.assertEqual(adapter.volume, 100)


if __name__ == "__main__":
    unittest.main()
