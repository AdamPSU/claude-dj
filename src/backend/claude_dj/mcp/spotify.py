from __future__ import annotations

import asyncio
import base64
import http.client
import json as json_module
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import HTTPSHandler, Request, build_opener, urlopen as _stdlib_urlopen

from .playback import SpotifyDevice, SpotifyPlaybackState, SpotifyPlaylist, Track


@dataclass(frozen=True)
class SpotifyConfig:
    client_id: str
    client_secret: str
    refresh_token: str
    api_base_url: str = "https://api.spotify.com/v1"
    token_url: str = "https://accounts.spotify.com/api/token"
    request_timeout_seconds: float = 10.0


class SpotifyWebAPIPlayer:
    def __init__(self, config: SpotifyConfig, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self.monotonic = monotonic
        self._access_token: str | None = None
        self._access_token_expires_at = 0.0

    async def start_track(self, spotify_uri: str) -> None:
        await self._api_request(
            "PUT",
            "/me/player/play",
            body={"uris": [spotify_uri]},
        )

    async def set_playback_volume(self, volume_percent: int) -> None:
        volume = max(0, min(100, int(volume_percent)))
        await self._api_request(
            "PUT",
            f"/me/player/volume?{urlencode({'volume_percent': volume})}",
        )

    async def pause_playback(self) -> None:
        await self._api_request("PUT", "/me/player/pause")

    async def resume_playback(self) -> None:
        await self._api_request("PUT", "/me/player/play")

    async def get_current_playback(self) -> SpotifyPlaybackState | None:
        response = await self._api_request("GET", "/me/player")
        if not response:
            return None

        item = response.get("item") or {}
        device = response.get("device") or None
        return SpotifyPlaybackState(
            track_id=item.get("id"),
            spotify_uri=item.get("uri"),
            progress_ms=response.get("progress_ms"),
            duration_ms=item.get("duration_ms"),
            is_playing=bool(response.get("is_playing")),
            device=SpotifyDevice(
                id=device.get("id"),
                name=device.get("name", "Spotify device"),
                volume_percent=device.get("volume_percent"),
                type=device.get("type"),
                is_active=bool(device.get("is_active")),
                is_restricted=bool(device.get("is_restricted")),
            )
            if device
            else None,
        )

    async def search_tracks(self, query: str, limit: int = 6) -> list[Track]:
        response = await self._api_request(
            "GET",
            f"/search?{urlencode({'q': query, 'type': 'track', 'limit': max(1, min(limit, 10))})}",
        )
        items = ((response or {}).get("tracks") or {}).get("items") or []
        return [
            track
            for item in items
            if (track := self._track_from_item(item, cluster=f"spotify_search:{query}")) is not None
        ]

    async def list_user_playlists(self, limit: int = 20) -> list[SpotifyPlaylist]:
        response = await self._api_request("GET", f"/me/playlists?{urlencode({'limit': max(1, min(limit, 50))})}")
        return [self._playlist_from_item(item) for item in (response or {}).get("items") or []]

    async def list_playlist_tracks(self, playlist_id: str, playlist_name: str, limit: int = 100) -> list[Track]:
        response = await self._api_request(
            "GET",
            f"/playlists/{playlist_id}/items?{urlencode({'limit': max(1, min(limit, 100))})}",
        )
        tracks: list[Track] = []
        for item in (response or {}).get("items") or []:
            track = self._track_from_item(item.get("track") or {}, cluster=f"playlist:{playlist_name}")
            if track is not None:
                tracks.append(track)
        return tracks

    async def list_devices(self) -> list[SpotifyDevice]:
        response = await self._api_request("GET", "/me/player/devices")
        return [self._device_from_item(item) for item in (response or {}).get("devices") or []]

    async def transfer_playback(self, device_id: str, *, play: bool = False) -> None:
        await self._api_request(
            "PUT",
            "/me/player",
            body={"device_ids": [device_id], "play": play},
        )

    async def _api_request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any] | None:
        token = await self._access_token_value()
        url = f"{self.config.api_base_url}{path}"
        headers = {"Authorization": f"Bearer {token}"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json_module.dumps(body).encode("utf-8")
        return await asyncio.to_thread(self._request_json_sync, method, url, headers, data)

    async def _access_token_value(self) -> str:
        if self._access_token is None or self.monotonic() >= self._access_token_expires_at:
            token, expires_in = await asyncio.to_thread(self._refresh_access_token_sync)
            self._access_token = token
            self._access_token_expires_at = self.monotonic() + max(0, expires_in - 60)
        return self._access_token

    def _refresh_access_token_sync(self) -> tuple[str, int]:
        credentials = f"{self.config.client_id}:{self.config.client_secret}".encode("utf-8")
        auth = base64.b64encode(credentials).decode("ascii")
        body = urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": self.config.refresh_token,
            }
        ).encode("utf-8")
        request = Request(
            self.config.token_url,
            data=body,
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.config.request_timeout_seconds) as response:
            payload = json_module.loads(response.read().decode("utf-8"))
        return payload["access_token"], int(payload.get("expires_in") or 3600)

    def _request_json_sync(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: bytes | None,
    ) -> dict[str, Any] | None:
        request = Request(url, data=data, headers=headers, method=method)
        with urlopen(request, timeout=self.config.request_timeout_seconds) as response:
            raw = response.read()
        if not raw:
            return None
        try:
            return json_module.loads(raw.decode("utf-8"))
        except json_module.JSONDecodeError:
            return None

    def _playlist_from_item(self, item: dict[str, Any]) -> SpotifyPlaylist:
        tracks = item.get("tracks") or {}
        return SpotifyPlaylist(
            id=item.get("id", ""),
            name=item.get("name", "Spotify Playlist"),
            public=item.get("public"),
            collaborative=bool(item.get("collaborative")),
            total_tracks=int(tracks.get("total") or 0),
        )

    def _track_from_item(self, item: dict[str, Any], *, cluster: str) -> Track | None:
        if item.get("is_local"):
            return None
        track_id = item.get("id")
        spotify_uri = item.get("uri")
        if not track_id or not spotify_uri:
            return None
        artists = item.get("artists") or []
        artist = ", ".join(artist_item.get("name", "") for artist_item in artists if artist_item.get("name"))
        album = item.get("album") or {}
        images = album.get("images") or []
        artwork_url = images[0].get("url", "") if images else ""
        return Track(
            id=track_id,
            title=item.get("name", "Spotify Track"),
            artist=artist or "Unknown Artist",
            spotify_uri=spotify_uri,
            cluster=cluster,
            duration_ms=int(item.get("duration_ms") or 180_000),
            artwork_url=artwork_url,
        )

    def _device_from_item(self, item: dict[str, Any]) -> SpotifyDevice:
        return SpotifyDevice(
            id=item.get("id"),
            name=item.get("name", "Spotify device"),
            volume_percent=item.get("volume_percent"),
            type=item.get("type"),
            is_active=bool(item.get("is_active")),
            is_restricted=bool(item.get("is_restricted")),
        )


def create_ipv4_connection(
    address,
    timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
    source_address=None,
):
    host, port = address
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in socket.getaddrinfo(
        host,
        port,
        socket.AF_INET,
        socket.SOCK_STREAM,
    ):
        sock = socket.socket(family, socktype, proto)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    if last_error is not None:
        raise last_error
    raise OSError(f"no IPv4 address found for {host}:{port}")


class IPv4HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = create_ipv4_connection


class IPv4HTTPSHandler(HTTPSHandler):
    def https_open(self, request):
        return self.do_open(IPv4HTTPSConnection, request, context=self._context)


def urlopen(request, timeout=None):
    if getattr(request, "type", "") == "https":
        return build_opener(IPv4HTTPSHandler()).open(request, timeout=timeout)
    return _stdlib_urlopen(request, timeout=timeout)
