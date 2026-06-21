"""Phase 1 tests — offline. The HTTP layer is mocked (token + paginated JSON).

Covers: pagination via ``next``, null/non-track skipping, empty-ISRC handling,
ISRC uppercasing, playlist-id normalization, and that every produced row
validates with ``validate_raw_track``.
"""

from __future__ import annotations

import json

import pytest

from recommendation_engine import scrape_spotify
from recommendation_engine.contracts import load_raw_tracks, validate_raw_track


# --- Fake HTTP session -------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    """Serves a refresh-token POST and a sequence of paginated GET pages."""

    def __init__(self, pages: list[dict]):
        self._pages = pages
        self.get_urls: list[str] = []

    def post(self, url, data=None, headers=None, timeout=None):  # noqa: ANN001
        return _FakeResponse({"access_token": "fake-access-token"})

    def get(self, url, headers=None, timeout=None):  # noqa: ANN001
        self.get_urls.append(url)
        # Page index is driven by call count; `next` URLs are opaque markers.
        idx = len(self.get_urls) - 1
        return _FakeResponse(self._pages[idx])


def _track(spotify_id="abc123", name="Song", artist="Artist",
           isrc="usrc12345678", album="Album", **extra):
    track = {
        "id": spotify_id,
        "name": name,
        "artists": [{"name": artist}],
        "external_ids": {"isrc": isrc} if isrc is not None else {},
        "album": {"name": album},
    }
    track.update(extra)
    # Spotify's /playlists/{id}/items keys the track object as "item".
    return {"item": track}


# --- normalize_playlist_id ---------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2tZuU4abcDEF", "2tZuU4abcDEF"),
        ("2tZuU4abcDEF?si=xyz", "2tZuU4abcDEF"),
        ("https://open.spotify.com/playlist/2tZuU4abcDEF?si=xyz", "2tZuU4abcDEF"),
        ("spotify:playlist:2tZuU4abcDEF", "2tZuU4abcDEF"),
        ("  2tZuU4abcDEF  ", "2tZuU4abcDEF"),
    ],
)
def test_normalize_playlist_id(raw, expected):
    assert scrape_spotify.normalize_playlist_id(raw) == expected


# --- extract_tracks ----------------------------------------------------------
def test_extract_skips_null_and_nontrack_items():
    items = [
        _track(spotify_id="t1"),
        {"item": None},                        # local/unavailable -> null
        {"item": {"type": "episode", "id": "e1", "name": "Pod",
                  "artists": [{"name": "Show"}]}},  # podcast episode
        {},                                    # malformed item
        _track(spotify_id="t2"),
    ]
    rows, skipped, empty_isrc = scrape_spotify.extract_tracks(items)
    assert [r.spotify_id for r in rows] == ["t1", "t2"]
    assert skipped == 3
    assert empty_isrc == 0


def test_isrc_uppercased_and_empty_counted():
    items = [
        _track(spotify_id="t1", isrc="gbduw0000059"),  # lowercase -> upper
        _track(spotify_id="t2", isrc=None),            # missing external_ids.isrc
        _track(spotify_id="t3", isrc=""),              # explicit empty
    ]
    rows, skipped, empty_isrc = scrape_spotify.extract_tracks(items)
    assert skipped == 0
    assert rows[0].isrc == "GBDUW0000059"
    assert rows[1].isrc == ""
    assert rows[2].isrc == ""
    assert empty_isrc == 2


def test_missing_artist_or_id_is_skipped():
    items = [
        {"item": {"id": "t1", "name": "X", "artists": []}},        # no artist
        {"item": {"id": None, "name": "Y", "artists": [{"name": "A"}]}},  # no id
        {"item": {"id": "t3", "name": None, "artists": [{"name": "A"}]}},  # no name
    ]
    rows, skipped, empty_isrc = scrape_spotify.extract_tracks(items)
    assert rows == []
    assert skipped == 3


def test_extracted_rows_validate_against_artifact_a():
    items = [_track(spotify_id="t1"), _track(spotify_id="t2", isrc=None)]
    rows, _, _ = scrape_spotify.extract_tracks(items)
    for r in rows:
        validate_raw_track(r.to_dict())  # raises if invalid


# --- scrape_playlist pagination ----------------------------------------------
def test_scrape_playlist_follows_next_until_null():
    pages = [
        {"items": [_track(spotify_id="t1"), _track(spotify_id="t2")],
         "next": "https://api.spotify.com/v1/...offset=100"},
        {"items": [_track(spotify_id="t3")], "next": None},
    ]
    session = _FakeSession(pages)
    rows, skipped, empty_isrc = scrape_spotify.scrape_playlist(
        "https://open.spotify.com/playlist/PID?si=abc",
        "fake-access-token",
        session=session,
    )
    assert [r.spotify_id for r in rows] == ["t1", "t2", "t3"]
    assert len(session.get_urls) == 2  # exactly two pages fetched
    # First request hit the normalized playlist id.
    assert "/playlists/PID/items" in session.get_urls[0]
    assert skipped == 0
    assert empty_isrc == 0


def test_scrape_playlists_merges_and_dedupes_by_spotify_id():
    pages = [
        {"items": [_track(spotify_id="t1"), _track(spotify_id="t2")], "next": None},
        {"items": [_track(spotify_id="t2", name="Duplicate"), _track(spotify_id="t3")], "next": None},
    ]
    session = _FakeSession(pages)

    rows, stats = scrape_spotify.scrape_playlists(["playlist-one", "playlist-two"], "fake-access-token", session=session)

    assert [row.spotify_id for row in rows] == ["t1", "t2", "t3"]
    assert stats["playlist_count"] == 2
    assert stats["duplicate_count"] == 1
    assert "/playlists/playlist-one/items" in session.get_urls[0]
    assert "/playlists/playlist-two/items" in session.get_urls[1]


def test_playlist_ids_from_env_prefers_plural_and_splits_commas_and_newlines(monkeypatch):
    env = {
        "SPOTIFY_PLAYLIST_ID": "single",
        "SPOTIFY_PLAYLIST_IDS": "https://open.spotify.com/playlist/one?si=abc, spotify:playlist:two\nthree",
    }
    monkeypatch.setattr(
        scrape_spotify.config,
        "getenv",
        lambda name, default=None, *, required=False: env.get(name, default),
    )

    assert scrape_spotify.playlist_ids_from_env() == ["one", "two", "three"]


def test_get_access_token_uses_refresh_flow():
    session = _FakeSession([])
    token = scrape_spotify.get_access_token(
        "cid", "secret", "refresh-tok", session=session
    )
    assert token == "fake-access-token"


# --- end-to-end run writes a valid Artifact A --------------------------------
def test_run_writes_valid_artifact(tmp_path, monkeypatch):
    pages = [
        {"items": [
            _track(spotify_id="t1", isrc="usrc11111111"),
            {"track": None},                       # skipped
            _track(spotify_id="t2", isrc=None),    # empty isrc
        ], "next": None},
        {"items": [_track(spotify_id="t2"), _track(spotify_id="t3")], "next": None},
    ]
    session = _FakeSession(pages)

    monkeypatch.setattr(scrape_spotify.requests, "Session", lambda: session)
    monkeypatch.setattr(scrape_spotify.config, "load_dotenv", lambda *a, **k: None)
    env = {
        "SPOTIFY_CLIENT_ID": "cid",
        "SPOTIFY_CLIENT_SECRET": "secret",
        "SPOTIFY_PLAYLIST_IDS": "PID-one,PID-two",
        "SPOTIFY_REFRESH_TOKEN": "refresh-tok",
    }
    monkeypatch.setattr(
        scrape_spotify.config,
        "getenv",
        lambda name, default=None, *, required=False: env.get(name, default),
    )

    out = tmp_path / "tracks_raw.json"
    rows = scrape_spotify.run(out_path=out)

    assert [r.spotify_id for r in rows] == ["t1", "t2", "t3"]
    assert len(session.get_urls) == 2
    # File on disk is a valid Artifact A.
    loaded = load_raw_tracks(out)
    assert [r.spotify_id for r in loaded] == ["t1", "t2", "t3"]
    assert loaded[0].isrc == "USRC11111111"
    assert loaded[1].isrc == ""
    # Sanity: raw JSON is an array.
    assert isinstance(json.loads(out.read_text()), list)
