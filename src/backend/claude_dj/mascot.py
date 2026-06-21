from __future__ import annotations

import subprocess
from pathlib import Path


class MascotAppProcess:
    def __init__(self, *, frontend_dir: Path | None = None) -> None:
        self.frontend_dir = frontend_dir or default_frontend_dir()
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        launcher = self.frontend_dir / "electron" / "launch.cjs"
        self.process = subprocess.Popen(
            ["node", str(launcher)],
            cwd=self.frontend_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def default_frontend_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "frontend"
