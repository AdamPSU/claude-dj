from __future__ import annotations

import asyncio
import json as json_module
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..observability import observe_async


@dataclass(frozen=True)
class DeepgramResponse:
    audio: bytes
    content_type: str


@dataclass(frozen=True)
class NarrationAudio:
    id: str
    text: str
    audio: bytes
    content_type: str
    model: str


class DeepgramRequester(Protocol):
    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, str],
    ) -> DeepgramResponse: ...


class Narrator(Protocol):
    async def generate(self, text: str) -> NarrationAudio: ...


class EphemeralNarrationStore:
    def __init__(self) -> None:
        self._next_id = 1
        self._audio: dict[str, NarrationAudio] = {}

    def save(self, *, text: str, audio: bytes, content_type: str, model: str) -> NarrationAudio:
        narration_id = f"narration-{self._next_id}"
        self._next_id += 1
        narration = NarrationAudio(
            id=narration_id,
            text=text,
            audio=audio,
            content_type=content_type,
            model=model,
        )
        self._audio[narration_id] = narration
        return narration

    def get(self, narration_id: str) -> NarrationAudio | None:
        return self._audio.get(narration_id)

    def delete(self, narration_id: str) -> None:
        self._audio.pop(narration_id, None)


class UrlLibDeepgramRequester:
    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, str],
    ) -> DeepgramResponse:
        return await asyncio.to_thread(self._post_sync, url, headers, json)

    def _post_sync(self, url: str, headers: dict[str, str], body: dict[str, str]) -> DeepgramResponse:
        request = Request(
            url,
            data=json_module.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request) as response:
            return DeepgramResponse(
                audio=response.read(),
                content_type=response.headers.get("Content-Type", "application/octet-stream"),
            )


class DeepgramNarrator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        speed: float | None = None,
        store: EphemeralNarrationStore,
        requester: DeepgramRequester | None = None,
        base_url: str = "https://api.deepgram.com/v1/speak",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.speed = speed
        self.store = store
        self.requester = requester or UrlLibDeepgramRequester()
        self.base_url = base_url

    async def generate(self, text: str) -> NarrationAudio:
        async def run() -> NarrationAudio:
            response = await self.requester.post(
                self._speak_url(),
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"text": text},
            )
            return self.store.save(
                text=text,
                audio=response.audio,
                content_type=response.content_type,
                model=self.model,
            )

        return await observe_async(
            "claude_dj.deepgram.generate_narration",
            op="http.client.deepgram",
            data={"model": self.model, "text_chars": len(text)},
            callback=run,
        )

    def _speak_url(self) -> str:
        query: dict[str, str] = {"model": self.model}
        if self.speed is not None:
            query["speed"] = f"{self.speed:g}"
        return f"{self.base_url}?{urlencode(query)}"
