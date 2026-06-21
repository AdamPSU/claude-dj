"""Emotion classification: interface, DeepFace impl, 3-bucket collapse, EMA.

DeepFace's 7-class output is noisy (happy/surprise confusion). We collapse
to 3 buckets:
  - positive: happy + surprise
  - neutral:  neutral
  - negative: angry + sad + fear + disgust

Then derive a single valence scalar (0 = negative, 0.5 = neutral, 1 = positive).

The classifier is behind an interface (EmotionClassifier) so the backbone
can be swapped later without touching downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vibe_dj import config

# Mapping from DeepFace 7-class keys to 3 buckets.
_RAW_TO_BUCKET: dict[str, str] = {
    "happy": "positive",
    "surprise": "positive",
    "neutral": "neutral",
    "angry": "negative",
    "sad": "negative",
    "fear": "negative",
    "disgust": "negative",
}

_BUCKET_KEYS = ("positive", "neutral", "negative")


# --- Pure functions (testable without DeepFace) ---


def collapse_to_buckets(raw_percentages: dict[str, float]) -> dict[str, float]:
    """Collapse 7-class percentages (0-100) into 3 normalized buckets."""
    buckets: dict[str, float] = {k: 0.0 for k in _BUCKET_KEYS}
    for emo, score in raw_percentages.items():
        bucket = _RAW_TO_BUCKET.get(emo.lower(), "negative")
        buckets[bucket] += score

    total = sum(buckets.values())
    if total > 0:
        buckets = {k: v / total for k, v in buckets.items()}
    return buckets


def ema_smooth(
    current: dict[str, float],
    previous: dict[str, float] | None,
    alpha: float = config.EMA_ALPHA,
) -> dict[str, float]:
    """Exponential moving average over the 3-bucket probabilities."""
    if previous is None:
        return current
    return {
        k: alpha * current.get(k, 0.0) + (1 - alpha) * previous.get(k, 0.0)
        for k in _BUCKET_KEYS
    }


def to_valence(probs: dict[str, float]) -> float:
    """Map 3-bucket probabilities to a single valence scalar in [0, 1].

    0 = fully negative, 0.5 = fully neutral, 1 = fully positive.
    """
    return 0.5 + 0.5 * (probs.get("positive", 0.0) - probs.get("negative", 0.0))


# --- Classifier interface ---


class EmotionClassifier(ABC):
    """Abstract classifier that returns 3-bucket probabilities."""

    @abstractmethod
    def classify(self, face_crop: np.ndarray) -> dict[str, float]:
        """Return {"positive": ..., "neutral": ..., "negative": ...} summing to ~1."""
        ...


class DeepFaceClassifier(EmotionClassifier):
    """Default classifier using DeepFace (wraps a FER-style CNN)."""

    def __init__(self):
        from deepface import DeepFace
        self._DeepFace = DeepFace

        # Warm up: first call downloads weights and is slow.
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        self._DeepFace.analyze(
            dummy, actions=["emotion"], enforce_detection=False,
            silent=True, detector_backend="skip",
        )

    def classify(self, face_crop: np.ndarray) -> dict[str, float]:
        try:
            results = self._DeepFace.analyze(
                face_crop, actions=["emotion"],
                enforce_detection=False, silent=True,
                detector_backend="skip",
            )
        except Exception:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

        if not results:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

        raw = results[0]["emotion"] if isinstance(results, list) else results["emotion"]
        return collapse_to_buckets(raw)
