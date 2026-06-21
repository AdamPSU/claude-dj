"""DJ Agent: threshold policy over (valence, vibe_score, motion_energy).

Start with transparent rules. Structured so a tiny classifier or an LLM
(Claude) can be dropped in later to reason over signal history.
"""

from __future__ import annotations

from dataclasses import dataclass

from vibe_dj import config


@dataclass
class AgentDecision:
    action: str       # "keep", "change_track", "increase_energy", "decrease_energy"
    reason: str
    confidence: float  # 0..1


class DJAgent:
    """Threshold-based DJ policy over three scalars."""

    def __init__(self):
        self._history: list[tuple[float, float, float]] = []

    def decide(
        self, valence: float, vibe_score: float, motion_energy: float,
    ) -> AgentDecision:
        self._history.append((valence, vibe_score, motion_energy))

        # Strong negative: listener disengaged -> change track
        if valence < config.VALENCE_NEGATIVE and vibe_score < config.VIBE_LOW:
            return AgentDecision(
                "change_track",
                "Listener disengaged (low valence + no vibe)",
                0.9,
            )

        # High vibe + positive valence: they're into it -> keep
        if vibe_score > config.VIBE_HIGH and valence > config.VALENCE_POSITIVE:
            return AgentDecision("keep", "Vibing and happy", 0.9)

        # High vibe but negative valence: catchy but wrong mood -> change
        if vibe_score > config.VIBE_HIGH and valence < config.VALENCE_NEGATIVE:
            return AgentDecision(
                "change_track",
                "Vibing but negative valence -- wrong mood",
                0.7,
            )

        # Low vibe + positive valence: enjoying but still -> calmer track
        if vibe_score < config.VIBE_LOW and valence > config.VALENCE_POSITIVE:
            return AgentDecision(
                "decrease_energy",
                "Happy but still -- try calmer track",
                0.5,
            )

        # High motion + low vibe: moving but not in sync -> increase energy
        if motion_energy > 0.6 and vibe_score < config.VIBE_LOW:
            return AgentDecision(
                "increase_energy",
                "Moving but not synced -- try higher energy",
                0.5,
            )

        # Default: neutral -> maintain
        return AgentDecision("keep", "Neutral signals -- maintaining", 0.3)

    def reset(self) -> None:
        self._history.clear()
