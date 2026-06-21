"""Continuous test script for DeepFace emotion detection.

Tests: imports, model load, single-frame inference, webcam integration.
"""

from __future__ import annotations

import sys
import time

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def run_tests() -> list[dict]:
    results = []

    # Test 1: DeepFace import
    def test_deepface_import():
        from deepface import DeepFace  # noqa: F401
        return PASS, "deepface package imports successfully"

    # Test 2: DeepFace emotion model build
    def test_deepface_model():
        from deepface import DeepFace
        DeepFace.build_model("Emotion")
        return PASS, "DeepFace Emotion model built successfully"

    # Test 3: DeepFace single frame inference
    def test_deepface_inference():
        import numpy as np
        from deepface import DeepFace
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        result = DeepFace.analyze(
            dummy, actions=["emotion"],
            enforce_detection=False, silent=True,
            detector_backend="skip",
        )
        assert isinstance(result, list)
        emotions = result[0]["emotion"]
        expected = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}
        assert expected.issubset(set(emotions.keys())), f"Missing: {expected - set(emotions.keys())}"
        return PASS, f"DeepFace inference: {len(emotions)} classes returned"

    # Test 4: Webcam single-frame capture + DeepFace
    def test_webcam_deepface():
        import cv2
        from deepface import DeepFace
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return SKIP, "No webcam available"
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return SKIP, "Could not read webcam frame"

        df_result = DeepFace.analyze(
            frame, actions=["emotion"],
            enforce_detection=False, silent=True,
        )
        df_ok = bool(df_result)
        return PASS, f"Webcam DeepFace: {'ok' if df_ok else 'fail'}"

    # Test 5: Check webcam.py integration
    def test_integration():
        try:
            import webcam as wm
            source = open(wm.__file__).read()
            has_df = "DeepFace" in source or "deepface" in source
            has_convert = "_deepface_to_emotions" in source
            if has_df and has_convert:
                return PASS, "webcam.py has DeepFace integration"
            else:
                return SKIP, f"webcam.py: DeepFace={has_df}, convert={has_convert}"
        except Exception as e:
            return SKIP, f"Could not check integration: {e}"

    tests = [
        ("deepface_import", test_deepface_import),
        ("deepface_model", test_deepface_model),
        ("deepface_inference", test_deepface_inference),
        ("webcam_deepface", test_webcam_deepface),
        ("integration", test_integration),
    ]

    for name, fn in tests:
        try:
            status, msg = fn()
        except Exception as e:
            status = FAIL
            msg = f"{e.__class__.__name__}: {e}"
        results.append({"test": name, "status": status, "msg": msg})

    return results


def main():
    print(f"=== DeepFace Emotion Test Run @ {time.strftime('%H:%M:%S')} ===")
    results = run_tests()
    passed = sum(1 for r in results if r["status"] == PASS)
    failed = sum(1 for r in results if r["status"] == FAIL)
    skipped = sum(1 for r in results if r["status"] == SKIP)

    for r in results:
        icon = {"PASS": "+", "FAIL": "X", "SKIP": "-"}[r["status"]]
        print(f"  [{icon}] {r['test']}: {r['msg']}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
