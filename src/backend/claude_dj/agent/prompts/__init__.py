from __future__ import annotations

from importlib.resources import files


PROMPT_FILENAMES = ("system.md", "on_start.md", "on_mid_song_prepare.md")


def load_prompt(filename: str) -> str:
    return files(__package__).joinpath(filename).read_text(encoding="utf-8")


DJ_SYSTEM_PROMPT = load_prompt("system.md")
START_HOOK_PROMPT = load_prompt("on_start.md")


def build_mid_song_prompt(*, progress_percent: int) -> str:
    return load_prompt("on_mid_song_prepare.md").format(progress_percent=progress_percent)
