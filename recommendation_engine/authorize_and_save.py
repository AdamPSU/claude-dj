"""One-shot Spotify OAuth over HTTPS loopback, write SPOTIFY_REFRESH_TOKEN to .env, self-test.

Spotify now requires an https redirect URI even for the 127.0.0.1 loopback, so this
runs a local HTTPS server with a self-signed cert and uses
    https://127.0.0.1:8888/callback
which must be registered as a Redirect URI on the app at developer.spotify.com.

Run interactively (opens a browser; you must click through a one-time
"your connection is not private" warning because the cert is self-signed):

    uv run python authorize_and_save.py
"""
from __future__ import annotations

import base64
import http.server
import ssl
import subprocess
import tempfile
import urllib.parse
import webbrowser
from pathlib import Path

import requests

from recommendation_engine import config
from recommendation_engine.scrape_spotify import (
    OAUTH_SCOPES,
    SPOTIFY_AUTHORIZE_URL,
    _basic_auth_header,
)

ENV_PATH = Path(__file__).parent / ".env"
HOST, PORT = "127.0.0.1", 8888
REDIRECT_URI = f"https://{HOST}:{PORT}/callback"


def _read_env_pair() -> tuple[str, str]:
    env = {}
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env["SPOTIFY_CLIENT_ID"], env["SPOTIFY_CLIENT_SECRET"]


def _write_refresh_token(token: str) -> None:
    lines = ENV_PATH.read_text().splitlines()
    out, replaced = [], False
    for line in lines:
        if line.startswith("SPOTIFY_REFRESH_TOKEN="):
            out.append(f"SPOTIFY_REFRESH_TOKEN={token}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"SPOTIFY_REFRESH_TOKEN={token}")
    ENV_PATH.write_text("\n".join(out) + "\n")


def _make_self_signed_cert(dirpath: Path) -> tuple[Path, Path]:
    cert, key = dirpath / "cert.pem", dirpath / "key.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key), "-out", str(cert),
            "-days", "1", "-nodes",
            "-subj", "/CN=127.0.0.1",
            "-addext", "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


class _CodeCatcher(http.server.BaseHTTPRequestHandler):
    code: str | None = None

    def do_GET(self):  # noqa: N802
        qs = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(qs)
        if "code" in params:
            _CodeCatcher.code = params["code"][0]
            body = b"<h2>ClaudeDJ authorized. You can close this tab.</h2>"
        else:
            body = b"<h2>No code received. Check the URL.</h2>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):  # silence the default stderr logging
        pass


def main() -> int:
    client_id, client_secret = _read_env_pair()
    print(f"Using client_id ...{client_id[-6:]} from .env; redirect_uri={REDIRECT_URI}")

    consent_url = SPOTIFY_AUTHORIZE_URL + "?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "scope": OAUTH_SCOPES,
        }
    )

    with tempfile.TemporaryDirectory() as td:
        cert, key = _make_self_signed_cert(Path(td))
        httpd = http.server.HTTPServer((HOST, PORT), _CodeCatcher)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

        print("\nOpen this URL (and approve in your browser):\n", consent_url, "\n")
        print("NOTE: the browser will warn 'your connection is not private' because the")
        print("local cert is self-signed -> click Advanced / Proceed to 127.0.0.1.\n")
        try:
            webbrowser.open(consent_url)
        except Exception:  # noqa: BLE001
            pass

        print(f"Waiting for the Spotify redirect on {REDIRECT_URI} ...")
        while _CodeCatcher.code is None:
            httpd.handle_request()
        code = _CodeCatcher.code

    resp = requests.post(
        config.SPOTIFY_TOKEN_URL,
        data={"grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI},
        headers=_basic_auth_header(client_id, client_secret),
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"CODE EXCHANGE FAILED: {resp.status_code} {resp.text[:300]}")
        return 1
    refresh_token = resp.json().get("refresh_token")
    if not refresh_token:
        print("No refresh_token in response:", resp.text[:300])
        return 1

    _write_refresh_token(refresh_token)
    print(f"\nWrote SPOTIFY_REFRESH_TOKEN to {ENV_PATH} (len={len(refresh_token)}, starts {refresh_token[:4]})")

    # Self-test
    r = requests.post(
        config.SPOTIFY_TOKEN_URL,
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        headers=_basic_auth_header(client_id, client_secret),
        timeout=30,
    )
    if r.status_code == 200:
        print("SELF-TEST OK: refresh token works. Tell Claude 'done'.")
        return 0
    print(f"SELF-TEST FAILED: {r.status_code} {r.text[:200]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
