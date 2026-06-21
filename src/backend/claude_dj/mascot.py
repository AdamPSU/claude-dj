from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from .mcp.narration import NarrationAudio, NarrationPlayer


MascotState = str


class MascotAppProcess:
    def __init__(self, *, frontend_dir: Path | None = None) -> None:
        self.frontend_dir = frontend_dir or default_frontend_dir()
        self.control_path = Path(tempfile.gettempdir()) / f"claude-dj-mascot-{uuid.uuid4().hex}.json"
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        self.sleep()
        launcher = self.frontend_dir / "electron" / "launch.cjs"
        self.process = subprocess.Popen(
            ["node", str(launcher)],
            cwd=self.frontend_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "CLAUDE_DJ_MASCOT_CONTROL": str(self.control_path)},
        )

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.control_path.unlink(missing_ok=True)
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            self.control_path.unlink(missing_ok=True)
        self.control_path.unlink(missing_ok=True)

    def sleep(self) -> None:
        self._write_state("sleeping")

    def start_speaking(self) -> None:
        self._write_state("speaking")

    def stop_speaking(self) -> None:
        self._write_state("normal")

    def _write_state(self, state: MascotState) -> None:
        self.control_path.write_text(json.dumps({"state": state}), encoding="utf-8")


class MascotNarrationPlayer:
    def __init__(self, inner: NarrationPlayer, mascot: MascotAppProcess) -> None:
        self.inner = inner
        self.mascot = mascot

    def play(self, narration: NarrationAudio) -> None:
        self.mascot.start_speaking()
        try:
            self.inner.play(narration)
        finally:
            self.mascot.stop_speaking()


def default_frontend_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "frontend"
