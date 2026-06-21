from __future__ import annotations

from typing import Any, Protocol

from ..observability import observe_async
from ..transition import InMemoryTransitionStore, TransitionPlan
from .narration import NarrationPlayer, NoopNarrationPlayer, Narrator
from .playback import InMemoryPlaybackRuntime


class ReactionSource(Protocol):
    async def get_reaction_signal(self) -> dict[str, Any]: ...


class NeutralReactionSource:
    async def get_reaction_signal(self) -> dict[str, Any]:
        return {"trend": "neutral", "confidence": 0.0, "available": False, "stub": True}


class DJToolHandlers:
    def __init__(
        self,
        transition_store: InMemoryTransitionStore,
        narrator: Narrator,
        playback: InMemoryPlaybackRuntime | None = None,
        narration_player: NarrationPlayer | None = None,
        reaction_source: ReactionSource | None = None,
    ) -> None:
        self.transition_store = transition_store
        self.narrator = narrator
        self.playback = playback or InMemoryPlaybackRuntime()
        self.narration_player = narration_player or NoopNarrationPlayer()
        self.reaction_source = reaction_source or NeutralReactionSource()

    async def get_session_context(self) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return await self.playback.get_session_context()

        return await self._observe_tool("get_session_context", {}, run)

    async def search_track_embeddings(
        self,
        query: str | None = None,
        mode: str = "text",
        seed_track_id: str | None = None,
        signal: str | None = None,
        avoid_clusters: list[str] | None = None,
        exclude_recent: bool = False,
        exclude_track_ids: list[str] | None = None,
        limit: int = 6,
    ) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return await self.playback.search_track_embeddings(
                query=query,
                mode=mode,
                seed_track_id=seed_track_id,
                signal=signal,
                avoid_clusters=avoid_clusters,
                exclude_recent=exclude_recent,
                exclude_track_ids=exclude_track_ids,
                limit=limit,
            )

        return await self._observe_tool(
            "search_track_embeddings",
            {
                "mode": mode,
                "query_chars": len(query or ""),
                "has_seed_track_id": seed_track_id is not None,
                "signal": signal or "",
                "avoid_cluster_count": len(avoid_clusters or []),
                "exclude_recent": exclude_recent,
                "exclude_track_count": len(exclude_track_ids or []),
                "limit": limit,
            },
            run,
        )

    async def get_seed_candidates(self, limit: int = 12, avoid_clusters: list[str] | None = None) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return await self.playback.get_seed_candidates(limit=limit, avoid_clusters=avoid_clusters)

        return await self._observe_tool(
            "get_seed_candidates",
            {"limit": limit, "avoid_cluster_count": len(avoid_clusters or [])},
            run,
        )

    async def replace_queue(
        self,
        track_ids: list[str],
        reason: str,
        timing: str = "now",
    ) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return await self.playback.replace_queue(track_ids, reason=reason, timing=timing)

        return await self._observe_tool(
            "replace_queue",
            {"track_count": len(track_ids), "reason": reason, "timing": timing},
            run,
        )

    async def narrate(
        self,
        text: str,
        mode: str,
        reason: str,
        timing: str | None = None,
        current_track_id: str | None = None,
        next_track_id: str | None = None,
        track_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            if mode == "immediate":
                audio = await self.narrator.generate(text)
                self.narration_player.play(audio)
                return {
                    "displayed": True,
                    "spoken": True,
                    "played": True,
                    "audio_id": audio.id,
                    "content_type": audio.content_type,
                    "model": audio.model,
                    "text": text,
                    "reason": reason,
                }
            if mode != "prepare":
                raise ValueError(f"unsupported narration mode: {mode}")
            if not current_track_id or not next_track_id or not track_ids:
                raise ValueError("prepared narration requires current_track_id, next_track_id, and track_ids")

            audio = await self.narrator.generate(text)
            self.transition_store.save(
                TransitionPlan(
                    current_track_id=current_track_id,
                    next_track_id=next_track_id,
                    track_ids=list(track_ids),
                    narration_id=audio.id,
                )
            )
            return {
                "audio_id": audio.id,
                "ready": True,
                "spoken": True,
                "content_type": audio.content_type,
                "model": audio.model,
                "text": text,
                "reason": reason,
                "timing": timing,
            }

        return await self._observe_tool(
            "narrate",
            {
                "mode": mode,
                "reason": reason,
                "timing": timing or "",
                "text_chars": len(text),
                "track_count": len(track_ids or []),
                "has_current_track_id": current_track_id is not None,
                "has_next_track_id": next_track_id is not None,
            },
            run,
        )

    async def play_track(self, track_id: str) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return await self.playback.play_track(track_id)

        return await self._observe_tool("play_track", {"track_id": track_id}, run)

    async def get_current_playback(self) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return await self.playback.get_current_playback()

        return await self._observe_tool("get_current_playback", {}, run)

    async def get_reaction_signal(self) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return await self.reaction_source.get_reaction_signal()

        return await self._observe_tool("get_reaction_signal", {}, run)

    async def mark_track_feedback(self, track_id: str, feedback: str, score: float | None = None) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return {
                "accepted": True,
                "stub": True,
                "track_id": track_id,
                "feedback": feedback,
                "score": score,
            }

        return await self._observe_tool(
            "mark_track_feedback",
            {"track_id": track_id, "feedback": feedback, "score": score if score is not None else ""},
            run,
        )

    async def summarize_session(self, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return {"accepted": True, "stub": True, "summary": summary or {}}

        return await self._observe_tool("summarize_session", {"summary_keys": sorted((summary or {}).keys())}, run)

    async def search_session_history(self, query: str, limit: int = 5) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return {"stub": True, "query": query, "limit": limit, "results": []}

        return await self._observe_tool(
            "search_session_history",
            {"query_chars": len(query), "limit": limit},
            run,
        )

    async def _observe_tool(
        self,
        tool_name: str,
        data: dict[str, Any],
        callback,
    ) -> dict[str, Any]:
        return await observe_async(
            f"claude_dj.mcp.{tool_name}",
            op="mcp.tool",
            data={"tool": tool_name, **data},
            callback=callback,
        )
