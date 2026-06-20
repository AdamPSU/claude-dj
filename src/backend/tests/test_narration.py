import unittest

from claude_dj.mcp.handlers import DJToolHandlers
from claude_dj.mcp.narration import DeepgramResponse, DeepgramNarrator, EphemeralNarrationStore, NarrationAudio
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


class FakeBoundaryAdapter:
    def __init__(self) -> None:
        self.volume = 100
        self.volume_events: list[int] = []
        self.played_tracks: list[str] = []
        self.played_narrations: list[str] = []

    async def get_music_volume(self) -> int:
        return self.volume

    async def set_music_volume(self, volume_percent: int) -> None:
        self.volume = volume_percent
        self.volume_events.append(volume_percent)

    async def play_track(self, track_id: str) -> None:
        self.played_tracks.append(track_id)

    async def play_narration(self, narration_id: str) -> None:
        self.played_narrations.append(narration_id)


class NarrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_deepgram_narrator_posts_text_and_stores_ephemeral_audio(self) -> None:
        store = EphemeralNarrationStore()
        requester = FakeRequester()
        narrator = DeepgramNarrator(
            api_key="dg-test-key",
            model="aura-2-apollo-en",
            speed=1.3,
            store=store,
            requester=requester,
        )

        audio = await narrator.generate("This groove is working, so I am moving one step over.")

        self.assertEqual(audio.id, "narration-1")
        self.assertEqual(audio.audio, b"wav-bytes")
        self.assertEqual(audio.content_type, "audio/wav")
        self.assertEqual(audio.model, "aura-2-apollo-en")
        self.assertIs(store.get("narration-1"), audio)
        self.assertEqual(
            requester.calls,
            [
                {
                    "url": "https://api.deepgram.com/v1/speak?model=aura-2-apollo-en&speed=1.3",
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
        handlers = DJToolHandlers(store, narrator)

        result = await handlers.narrate(
            text="I found a warm start. Let me open with this pocket.",
            mode="immediate",
            reason="startup",
        )

        self.assertEqual(result["audio_id"], "narration-1")
        self.assertTrue(result["displayed"])
        self.assertTrue(result["spoken"])
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
        store.save(
            TransitionPlan(
                current_track_id="track-a",
                next_track_id="track-b",
                track_ids=["track-b", "track-c", "track-d"],
                narration_id="narration-1",
            )
        )

        await BoundaryExecutor(store, adapter).on_track_boundary(ended_track_id="track-a")

        self.assertEqual(adapter.volume_events, [10, 100])
        self.assertEqual(adapter.played_tracks, ["track-b"])
        self.assertEqual(adapter.played_narrations, ["narration-1"])
        self.assertIsNone(store.get_ready_plan("track-a"))


if __name__ == "__main__":
    unittest.main()
