import json
import unittest
from unittest.mock import patch

from claude_dj.mcp.spotify import SpotifyConfig, SpotifyWebAPIPlayer


class FakeResponse:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        if self.payload is None:
            return b""
        return json.dumps(self.payload).encode("utf-8")


class SpotifyDeviceTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_devices_and_transfers_playback(self) -> None:
        responses = [
            FakeResponse({"access_token": "access-token-1"}),
            FakeResponse(
                {
                    "devices": [
                        {
                            "id": "device-1",
                            "name": "MacBook",
                            "type": "Computer",
                            "volume_percent": 80,
                            "is_active": False,
                            "is_restricted": False,
                        }
                    ]
                }
            ),
            FakeResponse(),
        ]
        requests = []

        def fake_urlopen(request, timeout=None):
            requests.append(request)
            return responses.pop(0)

        player = SpotifyWebAPIPlayer(
            SpotifyConfig(
                client_id="client-id",
                client_secret="client-secret",
                refresh_token="refresh-token",
            )
        )

        with patch("claude_dj.mcp.spotify.urlopen", fake_urlopen):
            devices = await player.list_devices()
            await player.transfer_playback(devices[0].id, play=False)

        self.assertEqual(devices[0].id, "device-1")
        self.assertEqual(devices[0].name, "MacBook")
        self.assertEqual(devices[0].type, "Computer")
        self.assertFalse(devices[0].is_active)
        self.assertFalse(devices[0].is_restricted)
        self.assertEqual(requests[1].full_url, "https://api.spotify.com/v1/me/player/devices")
        self.assertEqual(requests[2].full_url, "https://api.spotify.com/v1/me/player")
        self.assertEqual(
            json.loads(requests[2].data.decode("utf-8")),
            {"device_ids": ["device-1"], "play": False},
        )


if __name__ == "__main__":
    unittest.main()
