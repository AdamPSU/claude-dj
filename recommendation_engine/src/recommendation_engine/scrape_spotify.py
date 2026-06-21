"""Phase 1 — Spotify playlist -> Artifact A (``data/tracks_raw.json``).

Reads a public/user playlist's items from the Spotify Web API, extracts
``title / artist / isrc / album_name / spotify_id`` per Artifact A, skips
null/non-track items, and writes a validated ``tracks_raw.json``.

Auth flow = **Authorization Code (user OAuth)**, NOT Client-Credentials.
Verified 2026-06-20 (see ``contexts/api-recommendations.md``):
``GET /playlists/{id}/tracks`` returns 403 under Client-Credentials because it
has no user context; playlist items are only readable with a user token. We
therefore refresh a stored **refresh token** into a short-lived access token on
every run (no browser needed once the one-time consent is done).

One-time consent (interactive, run from the session so a browser can open)::

    uv run python -m recommendation_engine.scrape_spotify --authorize

That prints a consent URL, runs a tiny loopback server on
``http://127.0.0.1:8888/callback`` to capture ``?code=...``, exchanges it for a
refresh token, and tells you to store ``SPOTIFY_REFRESH_TOKEN`` in ``.env``.

Each run::

    uv run python -m recommendation_engine.scrape_spotify
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import requests

from recommendation_engine import config
from recommendation_engine.contracts import RawTrack, dump_json, validate_raw_track

logger = logging.getLogger("recommendation_engine.scrape_spotify")

# OAuth constants for the one-time Authorization Code consent helper.
OAUTH_SCOPES = "playlist-read-private playlist-read-collaborative"
OAUTH_REDIRECT_HOST = "127.0.0.1"  # Spotify rejects "localhost"; use the loopback IP.
OAUTH_REDIRECT_PORT = 8888
OAUTH_REDIRECT_URI = f"http://{OAUTH_REDIRECT_HOST}:{OAUTH_REDIRECT_PORT}/callback"
SPOTIFY_AUTHORIZE_URL = "https://accounts.spotify.com/authorize"

# Only the fields we actually need, to keep responses small.
# Spotify deprecated /playlists/{id}/tracks (returns 403 for Development-mode apps,
# Nov 2024). The replacement is /playlists/{id}/items, where each row's track object
# is keyed "item" (not "track"). See contexts/api-recommendations.md.
PLAYLIST_FIELDS = "next,items(item(id,name,artists(name),external_ids(isrc),album(name)))"
PAGE_LIMIT = 100
REQUEST_TIMEOUT = 30


# --- Auth --------------------------------------------------------------------
def _basic_auth_header(client_id: str, client_secret: str) -> dict[str, str]:
    raw = f"{client_id}:{client_secret}".encode()
    return {"Authorization": "Basic " + base64.b64encode(raw).decode()}


def get_access_token(
    client_id: str,
    client_secret: str,
    refresh_token: str,
    *,
    session: requests.Session | None = None,
) -> str:
    """Exchange the stored refresh token for a fresh access token (no browser)."""
    http = session or requests
    resp = http.post(
        config.SPOTIFY_TOKEN_URL,
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        headers=_basic_auth_header(client_id, client_secret),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("Spotify token refresh returned no access_token")
    return token


def normalize_playlist_id(raw: str) -> str:
    """Strip URL wrappers / query strings: ``2tZuU4...?si=abc`` -> ``2tZuU4...``.

    Accepts a bare id, an ``open.spotify.com/playlist/<id>`` URL, or a
    ``spotify:playlist:<id>`` URI.
    """
    value = raw.strip()
    if "spotify:playlist:" in value:
        value = value.split("spotify:playlist:", 1)[1]
    if "/playlist/" in value:
        value = value.split("/playlist/", 1)[1]
    # Drop any query string / fragment / trailing path segment.
    value = value.split("?", 1)[0].split("#", 1)[0].split("/", 1)[0]
    return value.strip()


# --- Extraction --------------------------------------------------------------
def _extract_row(track: dict[str, Any]) -> RawTrack | None:
    """Build a RawTrack from a Spotify ``track`` object, or None if unusable."""
    spotify_id = track.get("id")
    name = track.get("name")
    artists = track.get("artists") or []
    if not spotify_id or not name or not artists:
        return None
    artist = (artists[0] or {}).get("name")
    if not artist:
        return None

    isrc = ((track.get("external_ids") or {}).get("isrc") or "").upper()
    album_name = ((track.get("album") or {}).get("name") or "")

    return RawTrack(
        spotify_id=spotify_id,
        title=name,
        artist=artist,
        isrc=isrc,
        album_name=album_name,
    )


def _iter_playlist_pages(
    playlist_id: str,
    access_token: str,
    *,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Page through ``/playlists/{id}/items`` following ``next`` until null.

    Returns the concatenated list of raw ``items`` objects. (The legacy
    ``/tracks`` endpoint now 403s for Development-mode apps.)
    """
    http = session or requests
    headers = {"Authorization": f"Bearer {access_token}"}
    url: str | None = (
        f"{config.SPOTIFY_API_BASE}/playlists/{playlist_id}/items"
        f"?limit={PAGE_LIMIT}&fields={urllib.parse.quote(PLAYLIST_FIELDS, safe='(),')}"
    )
    items: list[dict[str, Any]] = []
    while url:
        resp = http.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        page = resp.json()
        items.extend(page.get("items") or [])
        url = page.get("next")
    return items


def extract_tracks(items: list[dict[str, Any]]) -> tuple[list[RawTrack], int, int]:
    """Turn raw playlist items into RawTracks.

    Returns ``(rows, skipped_count, empty_isrc_count)``. Null/non-track items
    (local files, unavailable tracks, podcast episodes) are skipped.
    """
    rows: list[RawTrack] = []
    skipped = 0
    empty_isrc = 0
    for item in items:
        # /items keys the track object as "item" (was "track" on the old /tracks endpoint).
        track = (item or {}).get("item")
        # Episodes have type "episode"; local/unavailable tracks come back null.
        if not isinstance(track, dict) or track.get("type") == "episode":
            skipped += 1
            continue
        row = _extract_row(track)
        if row is None:
            skipped += 1
            continue
        if row.isrc == "":
            empty_isrc += 1
        rows.append(row)
    return rows, skipped, empty_isrc


def scrape_playlist(
    playlist_id: str,
    access_token: str,
    *,
    session: requests.Session | None = None,
) -> tuple[list[RawTrack], int, int]:
    """Fetch + extract every track in a playlist."""
    pid = normalize_playlist_id(playlist_id)
    items = _iter_playlist_pages(pid, access_token, session=session)
    return extract_tracks(items)


# --- One-time OAuth consent helper ------------------------------------------
class _CodeCatcher(BaseHTTPRequestHandler):
    code: str | None = None

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        type(self).code = (params.get("code") or [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        msg = b"<h1>ClaudeDJ: authorization received. You can close this tab.</h1>"
        self.wfile.write(msg)

    def log_message(self, *_args: Any) -> None:  # silence default logging
        return


def authorize(client_id: str, client_secret: str) -> str:
    """Interactive one-time consent -> returns a refresh token to store in .env."""
    consent_url = SPOTIFY_AUTHORIZE_URL + "?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": OAUTH_REDIRECT_URI,
            "scope": OAUTH_SCOPES,
        }
    )
    print("Open this URL to authorize ClaudeDJ:\n", consent_url, "\n")
    try:
        webbrowser.open(consent_url)
    except Exception:  # noqa: BLE001 - headless is fine; URL is printed above
        pass

    server = HTTPServer((OAUTH_REDIRECT_HOST, OAUTH_REDIRECT_PORT), _CodeCatcher)
    print(f"Waiting for the Spotify redirect on {OAUTH_REDIRECT_URI} ...")
    while _CodeCatcher.code is None:
        server.handle_request()
    code = _CodeCatcher.code

    resp = requests.post(
        config.SPOTIFY_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": OAUTH_REDIRECT_URI,
        },
        headers=_basic_auth_header(client_id, client_secret),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    refresh_token = resp.json().get("refresh_token")
    if not refresh_token:
        raise RuntimeError("Spotify did not return a refresh_token")
    return refresh_token


# --- Entry point -------------------------------------------------------------
def run(out_path: Path | None = None) -> list[RawTrack]:
    """Full Phase 1 run: refresh token -> scrape -> validate -> write Artifact A."""
    config.load_dotenv()
    client_id = config.getenv("SPOTIFY_CLIENT_ID", required=True)
    client_secret = config.getenv("SPOTIFY_CLIENT_SECRET", required=True)
    playlist_id = config.getenv("SPOTIFY_PLAYLIST_ID", required=True)
    refresh_token = config.getenv("SPOTIFY_REFRESH_TOKEN", required=True)

    session = requests.Session()
    access_token = get_access_token(
        client_id, client_secret, refresh_token, session=session
    )
    rows, skipped, empty_isrc = scrape_playlist(
        playlist_id, access_token, session=session
    )

    # Validate every row against Artifact A before writing.
    payload = [validate_raw_track(r.to_dict()).to_dict() for r in rows]
    if not payload:
        raise RuntimeError("No valid tracks extracted from playlist")

    out = out_path or config.TRACKS_RAW_PATH
    dump_json(rows, out)

    logger.info(
        "scraped %d tracks (skipped %d non-track items, %d with empty ISRC) -> %s",
        len(rows),
        skipped,
        empty_isrc,
        out,
    )
    print(
        f"Wrote {len(rows)} tracks to {out} "
        f"(skipped {skipped} non-track items, {empty_isrc} with empty ISRC)"
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Spotify playlist -> tracks_raw.json")
    parser.add_argument(
        "--authorize",
        action="store_true",
        help="run the one-time interactive OAuth consent and print a refresh token",
    )
    args = parser.parse_args(argv)

    if args.authorize:
        config.load_dotenv()
        client_id = config.getenv("SPOTIFY_CLIENT_ID", required=True)
        client_secret = config.getenv("SPOTIFY_CLIENT_SECRET", required=True)
        token = authorize(client_id, client_secret)
        print("\nStore this in recommendation_engine/.env:\n")
        print(f"SPOTIFY_REFRESH_TOKEN={token}")
        return 0

    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
