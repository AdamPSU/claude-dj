"""Spotify implementation of :class:`HistoryProvider`.

Uses ``GET /me/player/recently-played?limit=1`` to read the single most recent
track. That endpoint requires the ``user-read-recently-played`` scope, which is
included in :data:`scrape_spotify.OAUTH_SCOPES`; a refresh token minted before
that scope was added must be re-authorized
(``uv run python authorize_and_save.py`` from ``recommendation_engine/``).

Token refresh reuses :func:`scrape_spotify.get_access_token` — no separate auth
code path lives here.
"""

from __future__ import annotations

from typing import Any

import requests

from .. import config
from ..scrape_spotify import REQUEST_TIMEOUT, get_access_token
from .provider import ExternalTrack

RECENTLY_PLAYED_URL = f"{config.SPOTIFY_API_BASE}/me/player/recently-played"


class SpotifyHistoryProvider:
    """Reads the user's last played track from the Spotify Web API."""

    name = "spotify"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.session = session or requests.Session()

    @classmethod
    def from_env(cls) -> SpotifyHistoryProvider:
        """Build from the same env vars the rest of the pipeline already uses."""
        config.load_dotenv()
        return cls(
            client_id=config.getenv("SPOTIFY_CLIENT_ID", required=True),
            client_secret=config.getenv("SPOTIFY_CLIENT_SECRET", required=True),
            refresh_token=config.getenv("SPOTIFY_REFRESH_TOKEN", required=True),
        )

    def last_played(self) -> ExternalTrack | None:
        access_token = get_access_token(
            self.client_id,
            self.client_secret,
            self.refresh_token,
            session=self.session,
        )
        resp = self.session.get(
            RECENTLY_PLAYED_URL,
            params={"limit": 1},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("items") or []
        if not items:
            return None
        track = (items[0] or {}).get("track") or {}
        return self._to_external(track)

    @staticmethod
    def _to_external(track: dict[str, Any]) -> ExternalTrack | None:
        track_id = track.get("id")
        name = track.get("name")
        artists = track.get("artists") or []
        if not track_id or not name or not artists:
            return None
        artist = (artists[0] or {}).get("name")
        if not artist:
            return None
        isrc = ((track.get("external_ids") or {}).get("isrc") or "").upper()
        album_name = ((track.get("album") or {}).get("name") or "")
        return ExternalTrack(
            title=name,
            artist=artist,
            isrc=isrc,
            album_name=album_name,
            source="spotify",
            source_id=track_id,
        )
