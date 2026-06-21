from __future__ import annotations

import argparse
import asyncio
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from dotenv import load_dotenv

from .mcp.narration import DeepgramNarrator, EphemeralNarrationStore, NarrationAudio
from .mcp.playback import InMemoryPlaybackRuntime, SpotifyDevice, SpotifyPlayer
from .mcp.spotify import SpotifyConfig, SpotifyWebAPIPlayer


class AudioPlayer(Protocol):
    def play(self, narration: NarrationAudio) -> None: ...


@dataclass(frozen=True)
class SmokeResult:
    track_id: str
    track_title: str
    narration_id: str
    device_name: str | None
    is_playing: bool
    starting_query: str


@dataclass(frozen=True)
class StartingContext:
    query: str
    reason: str


class LocalAudioPlayer:
    def __init__(self, *, timeout_seconds: float = 20.0) -> None:
        self.timeout_seconds = timeout_seconds

    def play(self, narration: NarrationAudio) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("local narration playback currently requires macOS afplay")
        suffix = self._suffix_for_content_type(narration.content_type)
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(narration.audio)
                path = Path(temp_file.name)
            subprocess.run(["afplay", str(path)], check=True, timeout=self.timeout_seconds)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def _suffix_for_content_type(self, content_type: str) -> str:
        if "mpeg" in content_type or "mp3" in content_type:
            return ".mp3"
        if "wav" in content_type:
            return ".wav"
        if "ogg" in content_type:
            return ".ogg"
        return ".audio"


async def choose_starting_context(spotify: SpotifyPlayer, *, override_query: str | None = None) -> StartingContext:
    if override_query:
        return StartingContext(query=override_query, reason="manual override")

    playlists = await spotify.list_user_playlists(limit=5)
    playlist_names = [playlist.name for playlist in playlists if playlist.name]
    if playlist_names:
        selected_names = playlist_names[:3]
        return StartingContext(
            query=" ".join(selected_names),
            reason=f"your Spotify playlists: {', '.join(selected_names)}",
        )

    playback = await spotify.get_current_playback()
    if playback and playback.track_id:
        return StartingContext(
            query="fresh music discovery",
            reason="current Spotify playback",
        )

    return StartingContext(
        query="fresh music discovery",
        reason="no active playback or playlists were available",
    )


async def run_autonomous_demo(
    *,
    spotify: SpotifyPlayer,
    narrator: DeepgramNarrator,
    audio_player: AudioPlayer,
    output: TextIO,
    query: str | None = None,
    track_limit: int = 3,
) -> SmokeResult:
    device = await _ensure_spotify_device(spotify, output)
    starting_context = await choose_starting_context(spotify, override_query=query)
    _print(output, f"demo: autonomous start from {starting_context.reason}")

    runtime = InMemoryPlaybackRuntime(
        tracks=[],
        spotify=spotify,
        seed_vibe=starting_context.query,
        playlist_limit=5,
        playlist_track_limit=50,
    )
    search = await runtime.search_track_embeddings(query=starting_context.query, limit=track_limit)
    candidates = search.get("candidates", [])
    if not candidates:
        raise RuntimeError("Spotify returned no demo track candidates")

    _print(output, f"spotify: ok candidates {[candidate['title'] for candidate in candidates]}")
    track_ids = [candidate["id"] for candidate in candidates]
    await runtime.replace_queue(track_ids, reason="autonomous_demo_start")
    first = candidates[0]

    narration = await narrator.generate(
        "ClaudeDJ is starting on its own. "
        f"I read {starting_context.reason} and found {first['title']} by {first['artist']}."
    )
    _print(output, f"deepgram: ok generated {narration.content_type} bytes={len(narration.audio)}")
    audio_player.play(narration)
    _print(output, "deepgram: ok played narration")

    await runtime.play_track(first["id"])
    await asyncio.sleep(2)
    playback = await runtime.get_current_playback()
    current = playback.get("current_track") or first
    _print(output, f"spotify: ok started {current['title']}")
    _print(output, "demo: ok")

    return SmokeResult(
        track_id=first["id"],
        track_title=current["title"],
        narration_id=narration.id,
        device_name=(playback.get("device") or {}).get("name") or (device.name if device else None),
        is_playing=bool(playback.get("is_playing")),
        starting_query=starting_context.query,
    )


async def run_live_autonomous_demo(*, output: TextIO, query: str | None, track_limit: int) -> SmokeResult:
    load_dotenv(".env")
    _require_env(
        [
            "SPOTIFY_CLIENT_ID",
            "SPOTIFY_CLIENT_SECRET",
            "SPOTIFY_REFRESH_TOKEN",
            "DEEPGRAM_API_KEY",
        ]
    )
    narration_store = EphemeralNarrationStore()
    spotify = SpotifyWebAPIPlayer(
        SpotifyConfig(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
            refresh_token=os.environ["SPOTIFY_REFRESH_TOKEN"],
            request_timeout_seconds=float(os.environ.get("SPOTIFY_REQUEST_TIMEOUT_SECONDS", "10")),
        )
    )
    narrator = DeepgramNarrator(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        model=os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-apollo-en"),
        speed=float(os.environ.get("DEEPGRAM_TTS_SPEED", "1.3")),
        store=narration_store,
    )
    return await run_autonomous_demo(
        spotify=spotify,
        narrator=narrator,
        audio_player=LocalAudioPlayer(),
        output=output,
        query=query,
        track_limit=track_limit,
    )


async def _ensure_spotify_device(spotify: SpotifyPlayer, output: TextIO) -> SpotifyDevice | None:
    playback = await spotify.get_current_playback()
    if playback and playback.device:
        _print(output, f"spotify: ok active device {playback.device.name}")
        return playback.device

    devices = await spotify.list_devices()
    candidates = [device for device in devices if device.id and not device.is_restricted]
    if not candidates:
        raise RuntimeError("No unrestricted Spotify Connect device found. Open Spotify on this Mac and retry.")

    device = next((candidate for candidate in candidates if candidate.is_active), candidates[0])
    if device.id and not device.is_active:
        await spotify.transfer_playback(device.id, play=False)
        _print(output, f"spotify: ok transferred device {device.name}")
    else:
        _print(output, f"spotify: ok active device {device.name}")
    return device


def _require_env(names: list[str]) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required env: {', '.join(missing)}")


def _print(output: TextIO, message: str) -> None:
    print(message, file=output, flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start an autonomous ClaudeDJ demo with Spotify + Deepgram.")
    parser.add_argument(
        "--query",
        default=os.environ.get("CLAUDE_DJ_DEMO_QUERY"),
        help="Optional manual override. Omit this for the real autonomous demo behavior.",
    )
    parser.add_argument("--track-limit", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        asyncio.run(run_live_autonomous_demo(output=sys.stdout, query=args.query, track_limit=args.track_limit))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"demo: failed {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
