import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from claude_dj.mascot import MascotAppProcess, MascotNarrationPlayer
from claude_dj.mcp.narration import NarrationAudio


class FakeProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


class FakePlayer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.played: list[NarrationAudio] = []

    def play(self, narration: NarrationAudio) -> None:
        self.played.append(narration)
        if self.fail:
            raise RuntimeError("playback failed")


class FakeMascot:
    def __init__(self) -> None:
        self.events: list[str] = []

    def start_speaking(self) -> None:
        self.events.append("speaking")

    def stop_speaking(self) -> None:
        self.events.append("normal")


class MascotTests(unittest.TestCase):
    def test_start_writes_sleeping_state_and_passes_control_file_to_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frontend_dir = Path(tmp)
            (frontend_dir / "electron").mkdir()
            process = FakeProcess()
            calls = []

            def fake_popen(args, **kwargs):
                calls.append({"args": args, **kwargs})
                return process

            with patch("claude_dj.mascot.subprocess.Popen", fake_popen):
                mascot = MascotAppProcess(frontend_dir=frontend_dir)
                mascot.start()

            self.assertEqual(json.loads(mascot.control_path.read_text())["state"], "sleeping")
            self.assertEqual(calls[0]["env"]["CLAUDE_DJ_MASCOT_CONTROL"], str(mascot.control_path))

    def test_narration_player_sets_speaking_only_while_audio_playback_runs(self) -> None:
        narration = NarrationAudio("narration-1", "hello", b"audio", "audio/wav", "model")
        mascot = FakeMascot()
        player = FakePlayer()

        MascotNarrationPlayer(player, mascot).play(narration)

        self.assertEqual(player.played, [narration])
        self.assertEqual(mascot.events, ["speaking", "normal"])

    def test_narration_player_restores_normal_state_when_playback_fails(self) -> None:
        narration = NarrationAudio("narration-1", "hello", b"audio", "audio/wav", "model")
        mascot = FakeMascot()
        player = FakePlayer(fail=True)

        with self.assertRaises(RuntimeError):
            MascotNarrationPlayer(player, mascot).play(narration)

        self.assertEqual(mascot.events, ["speaking", "normal"])


if __name__ == "__main__":
    unittest.main()
