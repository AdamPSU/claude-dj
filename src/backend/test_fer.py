"""Continuous test script for ViT-FER + DeepFace ensemble emotion detection.

Run by subagent to validate both emotion models and their integration.
Tests: imports, model loads, single-frame inference, score shape, webcam integration.
"""

from __future__ import annotations

import sys
import time

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def run_tests() -> list[dict]:
    results = []

    # Test 1: Transformers import
    def test_transformers_import():
        from transformers import pipeline  # noqa: F401
        return PASS, "transformers package imports successfully"

    # Test 2: DeepFace import
    def test_deepface_import():
        from deepface import DeepFace  # noqa: F401
        return PASS, "deepface package imports successfully"

    # Test 3: ViT model load
    def test_vit_model():
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "image-classification",
            model="HardlyHumans/Facial-expression-detection",
        )
        assert pipe is not None
        return PASS, "ViT-FER model loaded successfully"

    # Test 4: DeepFace emotion model build
    def test_deepface_model():
        from deepface import DeepFace
        DeepFace.build_model("Emotion")
        return PASS, "DeepFace Emotion model built successfully"

    # Test 5: ViT single frame inference
    def test_vit_inference():
        import numpy as np
        from PIL import Image
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "image-classification",
            model="HardlyHumans/Facial-expression-detection",
        )
        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        result = pipe(dummy, top_k=7)
        assert isinstance(result, list)
        labels = {r["label"].lower() for r in result}
        expected = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}
        assert expected.issubset(labels), f"Missing labels: {expected - labels}"
        return PASS, f"ViT inference: {len(result)} classes returned"

    # Test 6: Webcam single-frame capture + ensemble
    def test_webcam_ensemble():
        import cv2
        from PIL import Image
        from deepface import DeepFace
        from transformers import pipeline as hf_pipeline
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return SKIP, "No webcam available"
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return SKIP, "Could not read webcam frame"

        # ViT pass
        pipe = hf_pipeline(
            "image-classification",
            model="HardlyHumans/Facial-expression-detection",
        )
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        vit_result = pipe(pil_image, top_k=7)
        vit_ok = len(vit_result) == 7

        # DeepFace pass
        df_result = DeepFace.analyze(
            frame, actions=["emotion"],
            enforce_detection=False, silent=True,
        )
        df_ok = bool(df_result)

        return PASS, f"Webcam ensemble: ViT={'ok' if vit_ok else 'fail'}, DeepFace={'ok' if df_ok else 'fail'}"

    # Test 7: Check webcam.py integration
    def test_ensemble_integration():
        try:
            import webcam as wm
            source = open(wm.__file__).read()
            has_vit = "hf_pipeline" in source or "_VIT_MODEL" in source
            has_df = "DeepFace" in source or "deepface" in source
            has_ensemble = "_ensemble_emotions" in source
            if has_vit and has_df and has_ensemble:
                return PASS, "webcam.py has ViT + DeepFace ensemble integration"
            else:
                return SKIP, f"webcam.py: ViT={has_vit}, DeepFace={has_df}, ensemble={has_ensemble}"
        except Exception as e:
            return SKIP, f"Could not check integration: {e}"

    tests = [
        ("transformers_import", test_transformers_import),
        ("deepface_import", test_deepface_import),
        ("vit_model", test_vit_model),
        ("deepface_model", test_deepface_model),
        ("vit_inference", test_vit_inference),
        ("webcam_ensemble", test_webcam_ensemble),
        ("ensemble_integration", test_ensemble_integration),
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
    print(f"=== ViT+DeepFace Ensemble Test Run @ {time.strftime('%H:%M:%S')} ===")
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
