"""Phase 2 acceptance: Deezer enrichment with a fully mocked HTTP layer.

No live network: every test drives a fake :class:`DeezerClient` whose
``get_json``/``download`` seams are backed by in-memory fixtures.
"""

from __future__ import annotations

import hashlib

import pytest

from recommendation_engine import config
from recommendation_engine.contracts import RawTrack, validate_enriched_track
from recommendation_engine.enrich_deezer import (
    DeezerClient,
    DeezerError,
    EnrichStats,
    RateLimiter,
    enrich_track,
    enrich_tracks,
)

PREVIEW_BYTES = b"ID3-fake-mp3-bytes"


def _deezer_track(deezer_id, album_id, *, preview=True, title="Song",
                  artist="Artist", album_title="Album"):
    return {
        "id": deezer_id,
        "title": title,
        "duration": 226,
        "rank": 814839,
        "preview": "https://cdn.deezer.com/preview.mp3" if preview else "",
        "artist": {"name": artist},
        "album": {"id": album_id, "title": album_title, "cover_xl": "https://art/xl.jpg"},
    }


class FakeClient(DeezerClient):
    """DeezerClient whose two network seams are served from dicts.

    ``isrc_map`` : isrc -> deezer track object (or DeezerError instance to raise)
    ``search_map``: query string -> list of track objects (search data array)
    ``album_map`` : str(album_id) -> album JSON object
    Counters record how many times each endpoint was hit.
    """

    def __init__(self, isrc_map=None, search_map=None, album_map=None):
        super().__init__(sleep=lambda _s: None)
        self.isrc_map = isrc_map or {}
        self.search_map = search_map or {}
        self.album_map = album_map or {}
        self.album_calls: dict[str, int] = {}
        self.isrc_calls = 0
        self.search_calls = 0
        self.downloads: list[str] = []

    def get_json(self, path, params=None):
        if path.startswith("/track/isrc:"):
            self.isrc_calls += 1
            isrc = path.split("/track/isrc:")[1]
            val = self.isrc_map.get(isrc)
            if isinstance(val, Exception):
                raise val
            if val is None:
                raise DeezerError("not found")
            return val
        if path == "/search":
            self.search_calls += 1
            data = self.search_map.get(params["q"], [])
            return {"data": data}
        if path.startswith("/album/"):
            album_id = path.split("/album/")[1]
            self.album_calls[album_id] = self.album_calls.get(album_id, 0) + 1
            album = self.album_map.get(album_id)
            if album is None:
                raise DeezerError("no album")
            return album
        raise AssertionError(f"unexpected path {path}")

    def download(self, url):
        self.downloads.append(url)
        return PREVIEW_BYTES


def _album(album_id, genre_name="Dance"):
    genres = {"data": [{"name": genre_name}]} if genre_name else {"data": []}
    return {"id": album_id, "title": "Album", "genres": genres}


# --- match_method: ISRC vs search --------------------------------------------
def test_match_method_isrc(tmp_path):
    raw = RawTrack("sp1", "Song", "Artist", "GBDUW0000059", "Album")
    client = FakeClient(
        isrc_map={"GBDUW0000059": _deezer_track(101, 9)},
        album_map={"9": _album(9, "Dance")},
    )
    stats = EnrichStats()
    row = enrich_track(raw, client, tmp_path, stats)
    assert row is not None
    assert row.match_method == "isrc"
    assert client.search_calls == 0
    assert row.deezer_id == "101"
    assert row.id == "deezer:101"


def test_match_method_search_fallback(tmp_path):
    # No ISRC -> should go straight to search.
    raw = RawTrack("sp1", "Song", "Artist", "", "Album")
    query = 'artist:"Artist" track:"Song"'
    client = FakeClient(
        search_map={query: [_deezer_track(202, 9)]},
        album_map={"9": _album(9, "Pop")},
    )
    stats = EnrichStats()
    row = enrich_track(raw, client, tmp_path, stats)
    assert row is not None
    assert row.match_method == "search"
    assert client.isrc_calls == 0
    assert client.search_calls == 1
    assert row.genre == "Pop"
    assert row.genre_tag == "pop"


def test_isrc_miss_falls_back_to_search(tmp_path):
    raw = RawTrack("sp1", "Song", "Artist", "BADISRC00000", "Album")
    query = 'artist:"Artist" track:"Song"'
    client = FakeClient(
        isrc_map={"BADISRC00000": DeezerError("not found")},
        search_map={query: [_deezer_track(303, 9)]},
        album_map={"9": _album(9, "Dance")},
    )
    stats = EnrichStats()
    row = enrich_track(raw, client, tmp_path, stats)
    assert row is not None
    assert row.match_method == "search"
    assert client.isrc_calls == 1
    assert client.search_calls == 1


# --- drop rules ---------------------------------------------------------------
def test_no_match_dropped(tmp_path):
    raw = RawTrack("sp1", "Song", "Artist", "", "Album")
    client = FakeClient(search_map={})  # empty search results
    stats = EnrichStats()
    row = enrich_track(raw, client, tmp_path, stats)
    assert row is None
    assert stats.dropped_no_match == 1
    assert client.downloads == []


def test_no_preview_dropped(tmp_path):
    raw = RawTrack("sp1", "Song", "Artist", "ISRC00000001", "Album")
    client = FakeClient(
        isrc_map={"ISRC00000001": _deezer_track(404, 9, preview=False)},
        album_map={"9": _album(9, "Dance")},
    )
    stats = EnrichStats()
    row = enrich_track(raw, client, tmp_path, stats)
    assert row is None
    assert stats.dropped_no_preview == 1
    assert client.downloads == []
    assert not list(tmp_path.iterdir())  # nothing downloaded


def test_no_genre_dropped(tmp_path):
    raw = RawTrack("sp1", "Song", "Artist", "ISRC00000002", "Album")
    client = FakeClient(
        isrc_map={"ISRC00000002": _deezer_track(505, 9)},
        album_map={"9": _album(9, genre_name="")},  # empty genres.data
    )
    stats = EnrichStats()
    row = enrich_track(raw, client, tmp_path, stats)
    assert row is None
    assert stats.dropped_no_genre == 1
    assert client.downloads == []  # no download for genre-less track


def test_album_missing_treated_as_no_genre(tmp_path):
    raw = RawTrack("sp1", "Song", "Artist", "ISRC00000003", "Album")
    client = FakeClient(
        isrc_map={"ISRC00000003": _deezer_track(606, 99)},
        album_map={},  # /album/99 raises -> no genre
    )
    stats = EnrichStats()
    row = enrich_track(raw, client, tmp_path, stats)
    assert row is None
    assert stats.dropped_no_genre == 1


# --- album cache --------------------------------------------------------------
def test_album_response_cached_across_tracks(tmp_path):
    # Three tracks share album id 7 -> /album/7 fetched exactly once.
    tracks = [
        RawTrack("sp1", "A", "Artist", "ISRCA0000001", "Album"),
        RawTrack("sp2", "B", "Artist", "ISRCB0000001", "Album"),
        RawTrack("sp3", "C", "Artist", "ISRCC0000001", "Album"),
    ]
    client = FakeClient(
        isrc_map={
            "ISRCA0000001": _deezer_track(11, 7, title="A"),
            "ISRCB0000001": _deezer_track(12, 7, title="B"),
            "ISRCC0000001": _deezer_track(13, 7, title="C"),
        },
        album_map={"7": _album(7, "Dance")},
    )
    rows, stats = enrich_tracks(tracks, client=client, audio_dir=tmp_path,
                               log=lambda _m: None)
    assert len(rows) == 3
    assert stats.enriched == 3
    assert client.album_calls["7"] == 1, "album should be fetched once and cached"


def test_no_genre_album_cached(tmp_path):
    # A genre-less album is cached too: second track does not re-fetch.
    tracks = [
        RawTrack("sp1", "A", "Artist", "ISRCD0000001", "Album"),
        RawTrack("sp2", "B", "Artist", "ISRCE0000001", "Album"),
    ]
    client = FakeClient(
        isrc_map={
            "ISRCD0000001": _deezer_track(21, 8, title="A"),
            "ISRCE0000001": _deezer_track(22, 8, title="B"),
        },
        album_map={"8": _album(8, genre_name="")},
    )
    rows, stats = enrich_tracks(tracks, client=client, audio_dir=tmp_path,
                               log=lambda _m: None)
    assert rows == []
    assert stats.dropped_no_genre == 2
    assert client.album_calls["8"] == 1


# --- surviving rows validate + side effects -----------------------------------
def test_surviving_row_validates_and_downloads(tmp_path):
    raw = RawTrack("spX", "Song", "Artist", "ISRCF0000001", "Album")
    client = FakeClient(
        isrc_map={"ISRCF0000001": _deezer_track(777, 9)},
        album_map={"9": _album(9, "Rap/Hip Hop")},
    )
    stats = EnrichStats()
    row = enrich_track(raw, client, tmp_path, stats)
    assert row is not None
    # Validates against the contract.
    validate_enriched_track(row.to_dict())
    # genre_tag slugified correctly.
    assert row.genre == "Rap/Hip Hop"
    assert row.genre_tag == "rap_hip_hop"
    # mp3 written and clip_hash matches the downloaded bytes.
    mp3 = tmp_path / "777.mp3"
    assert mp3.exists()
    assert mp3.read_bytes() == PREVIEW_BYTES
    assert row.clip_hash == hashlib.sha256(PREVIEW_BYTES).hexdigest()
    assert row.preview_source == "deezer"
    assert row.duration_seconds == 226
    assert row.rank == 814839
    assert row.artwork_url == "https://art/xl.jpg"


# --- error-key handling -------------------------------------------------------
def test_http200_error_payload_raises():
    """An HTTP-200 {"error": ...} payload (non-quota) is raised, not returned."""
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"error": {"code": 800, "message": "no data"}}

    class _Session:
        def get(self, *a, **k):
            return _Resp()

    client = DeezerClient(session=_Session(), sleep=lambda _s: None)
    with pytest.raises(DeezerError):
        client.get_json("/track/isrc:WHATEVER")


def test_quota_error_retries_then_succeeds():
    """code==4 quota errors back off and retry; eventual success returns data."""
    calls = {"n": 0}

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class _Session:
        def get(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Resp({"error": {"code": 4, "message": "Quota limit exceeded"}})
            return _Resp({"id": 1, "title": "ok"})

    client = DeezerClient(session=_Session(), sleep=lambda _s: None)
    data = client.get_json("/track/isrc:X")
    assert data["title"] == "ok"
    assert calls["n"] == 2


def test_transient_5xx_retries():
    calls = {"n": 0}

    class _Resp:
        def __init__(self, status, payload=None):
            self.status_code = status
            self._payload = payload or {}

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class _Session:
        def get(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Resp(503)
            return _Resp(200, {"id": 5, "title": "ok"})

    client = DeezerClient(session=_Session(), sleep=lambda _s: None)
    data = client.get_json("/album/5")
    assert data["id"] == 5
    assert calls["n"] == 2


# --- rate limiter -------------------------------------------------------------
def test_rate_limiter_sleeps_when_window_full():
    slept = []
    now = [0.0]
    limiter = RateLimiter(
        max_requests=2, window_seconds=5,
        sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)),
        clock=lambda: now[0],
    )
    limiter.acquire()  # t=0
    limiter.acquire()  # t=0, window now full
    limiter.acquire()  # must sleep ~5s to free a slot
    assert slept, "limiter should have slept once the window filled"
    assert slept[0] == pytest.approx(5, abs=1e-6)
