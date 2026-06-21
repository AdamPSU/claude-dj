from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from . import config


RAW_TO_BUCKET: dict[str, str] = {
    "happy": "positive",
    "surprise": "positive",
    "neutral": "neutral",
    "angry": "negative",
    "sad": "negative",
    "fear": "negative",
    "disgust": "negative",
}
BUCKET_KEYS = ("positive", "neutral", "negative")


def collapse_to_buckets(raw_percentages: dict[str, float]) -> dict[str, float]:
    buckets = {key: 0.0 for key in BUCKET_KEYS}
    for emotion, score in raw_percentages.items():
        bucket = RAW_TO_BUCKET.get(emotion.lower(), "negative")
        buckets[bucket] += float(score)
    total = sum(buckets.values())
    if total > 0:
        buckets = {key: value / total for key, value in buckets.items()}
    return buckets


def ema_smooth(
    current: dict[str, float],
    previous: dict[str, float] | None,
    *,
    alpha: float = config.EMA_ALPHA,
) -> dict[str, float]:
    if previous is None:
        return dict(current)
    return {
        key: (alpha * current.get(key, 0.0)) + ((1.0 - alpha) * previous.get(key, 0.0))
        for key in BUCKET_KEYS
    }


def to_valence(probs: dict[str, float]) -> float:
    return 0.5 + 0.5 * (probs.get("positive", 0.0) - probs.get("negative", 0.0))


class EmotionClassifier(ABC):
    @abstractmethod
    def classify(self, face_crop: np.ndarray) -> dict[str, float]: ...


class DeepFaceClassifier(EmotionClassifier):
    def __init__(self) -> None:
        from deepface import DeepFace

        self._deepface: Any = DeepFace
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        self._deepface.analyze(
            dummy,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
            detector_backend="skip",
        )

    def classify(self, face_crop: np.ndarray) -> dict[str, float]:
        try:
            results = self._deepface.analyze(
                face_crop,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
                detector_backend="skip",
            )
        except Exception:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
        if not results:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
        raw = results[0]["emotion"] if isinstance(results, list) else results["emotion"]
        return collapse_to_buckets(raw)
