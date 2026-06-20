"""Continuous test script for FER (Facial Expression Recognition) integration.

Run by subagent to validate the FER model as it's implemented.
Tests: import, model load, single-frame inference, score shape, webcam integration.
"""

from __future__ import annotations

import sys
import time
import traceback

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def run_tests() -> list[dict]:
    results = []

    # Test 1: FER import
    def test_fer_import():
        from fer.fer import FER  # noqa: F401
        return PASS, "fer package imports successfully"

    # Test 2: FER detector instantiation
    def test_fer_detector():
        from fer.fer import FER
        detector = FER(mtcnn=True)
        assert detector is not None
        return PASS, f"FER detector created (type={type(detector).__name__})"

    # Test 3: Single frame inference with a dummy image
    def test_fer_inference():
        import numpy as np
        from fer.fer import FER
        detector = FER(mtcnn=False)  # mtcnn=False is faster for test
        # Create a simple test frame (black image — expect no face detected)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_emotions(dummy)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        return PASS, f"Inference returned {len(result)} detections on blank frame"

    # Test 4: Webcam single-frame capture + FER
    def test_fer_webcam_frame():
        import cv2
        from fer.fer import FER
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return SKIP, "No webcam available"
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return SKIP, "Could not read webcam frame"
        detector = FER(mtcnn=False)
        result = detector.detect_emotions(frame)
        assert isinstance(result, list)
        if result:
            emotions = result[0]["emotions"]
            assert isinstance(emotions, dict)
            expected_keys = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}
            assert expected_keys.issubset(emotions.keys()), f"Missing keys: {expected_keys - emotions.keys()}"
            return PASS, f"Webcam FER: {emotions}"
        return PASS, "Webcam FER: no face detected in frame (ok for test)"

    # Test 5: Check if reaction.py integration exists (user's upcoming work)
    def test_fer_reaction_integration():
        try:
            from reaction import ReactionFrame
            # Check if any module references FER
            import webcam as wm
            source = open(wm.__file__).read()
            if "fer" in source.lower() or "FER" in source:
                return PASS, "webcam.py references FER — integration detected"
            else:
                return SKIP, "webcam.py does not reference FER yet (waiting for implementation)"
        except Exception as e:
            return SKIP, f"Could not check integration: {e}"

    # Run all tests
    tests = [
        ("fer_import", test_fer_import),
        ("fer_detector", test_fer_detector),
        ("fer_inference", test_fer_inference),
        ("fer_webcam_frame", test_fer_webcam_frame),
        ("fer_reaction_integration", test_fer_reaction_integration),
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
    print(f"=== FER Test Run @ {time.strftime('%H:%M:%S')} ===")
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
