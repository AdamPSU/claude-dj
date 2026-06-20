"""Quick test script for face detection. Press 'q' to quit."""

import cv2
import mediapipe as mp
from pathlib import Path
from webcam import WebcamWorker

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
_MODEL_PATH = str(Path(__file__).parent / "face_landmarker.task")


def main():
    print("Starting webcam face detection test...")
    print("Press 'q' in the video window to quit.\n")

    worker = WebcamWorker(sample_interval=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    worker.start()
    print("Webcam worker started. Capturing baseline...\n")

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_image)

            # Draw landmarks on frame
            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    # Draw dots at each landmark
                    for lm in face_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

            # Show latest reaction data from worker
            recent = worker.get_recent_frames(n=1)
            if recent:
                f = recent[-1]
                status = (
                    f"presence={f.presence:.0f}  "
                    f"movement={f.movement or 0:.3f}  "
                    f"face={f.face or 0:.3f}"
                )
                cv2.putText(frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if f.dominant_emotion:
                    emotion_text = f"emotion: {f.dominant_emotion}"
                    cv2.putText(frame, emotion_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                if f.emotions:
                    y_offset = 90
                    for emo, val in sorted(f.emotions.items(), key=lambda x: -x[1]):
                        bar_len = int(val * 200)
                        pct = f"{val*100:.1f}%"
                        color = (0, 255, 0) if emo in ("happy", "surprise") else (0, 150, 255)
                        cv2.putText(frame, f"{emo[:3]}", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        cv2.rectangle(frame, (50, y_offset - 10), (50 + bar_len, y_offset), color, -1)
                        cv2.putText(frame, pct, (55 + bar_len, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        y_offset += 22

                if worker.baseline:
                    b = worker.baseline
                    baseline_text = f"baseline: movement={b.movement:.3f} face={b.face:.3f}"
                    cv2.putText(frame, baseline_text, (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                emo_str = ""
                if f.emotions:
                    emo_str = "  ".join(f"{k}={v*100:.0f}%" for k, v in sorted(f.emotions.items(), key=lambda x: -x[1]))
                print(f"\r{status}  [{f.dominant_emotion or 'none'}] {emo_str}      ", end="", flush=True)

            cv2.imshow("ClaudeDJ Face Detection Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    print("\n\nStopping...")
    worker.stop()
    cap.release()
    cv2.destroyAllWindows()

    all_frames = worker.get_all_frames()
    print(f"\nCaptured {len(all_frames)} reaction frames.")
    if worker.baseline:
        b = worker.baseline
        print(f"Baseline - movement: {b.movement:.3f}, face: {b.face:.3f}")


if __name__ == "__main__":
    main()
