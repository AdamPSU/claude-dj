from __future__ import annotations

from typing import Any

from ..observability import observe_async
from ..transition import InMemoryTransitionStore, TransitionPlan
from .narration import Narrator


class DJToolHandlers:
    def __init__(self, transition_store: InMemoryTransitionStore, narrator: Narrator) -> None:
        self.transition_store = transition_store
        self.narrator = narrator

    async def get_session_context(self) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return {
                "seed_vibe": "reggaeton",
                "current_track_id": "stub-current-track",
                "current_cluster": "stub-reggaeton",
                "queue_track_ids": [],
                "recent_track_ids": [],
                "recommended_next_action": "start_initial_set",
            }

        return await self._observe_tool("get_session_context", {}, run)

    async def search_track_embeddings(
        self,
        query: str | None = None,
        mode: str = "text",
        avoid_clusters: list[str] | None = None,
        limit: int = 6,
    ) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            capped_limit = max(3, min(limit, 6))
            cluster = "stub-shift" if mode in {"shift", "adjacent_shift", "slight_shift"} else "stub-reggaeton"
            return {
                "available": False,
                "stub": True,
                "query": query,
                "mode": mode,
                "avoid_clusters": avoid_clusters or [],
                "candidates": [
                    {
                        "id": f"stub-track-{index}",
                        "title": f"Stub Track {index}",
                        "artist": "ClaudeDJ Stub",
                        "cluster": cluster,
                        "score": round(1.0 - (index * 0.01), 2),
                    }
                    for index in range(1, capped_limit + 1)
                ],
            }

        return await self._observe_tool(
            "search_track_embeddings",
            {
                "mode": mode,
                "query_chars": len(query or ""),
                "avoid_cluster_count": len(avoid_clusters or []),
                "limit": limit,
            },
            run,
        )

    async def replace_queue(
        self,
        track_ids: list[str],
        reason: str,
        timing: str = "now",
    ) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return {
                "accepted": True,
                "stub": True,
                "track_ids": track_ids,
                "reason": reason,
                "timing": timing,
            }

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
                return {
                    "displayed": True,
                    "spoken": True,
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
            return {"started": True, "stub": True, "track_id": track_id}

        return await self._observe_tool("play_track", {"track_id": track_id}, run)

    async def get_current_playback(self) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return {
                "current_track_id": "stub-current-track",
                "current_cluster": "stub-reggaeton",
                "progress_percent": 55,
                "queue_track_ids": [],
            }

        return await self._observe_tool("get_current_playback", {}, run)

    async def get_reaction_signal(self) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return {"trend": "neutral", "confidence": 0.0, "available": False, "stub": True}

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
