"""Quick test script for face detection + FER emotions. Press 'q' to quit."""

import cv2
import mediapipe as mp
from fer.fer import FER as FERDetector
from pathlib import Path
from webcam import _fer_engagement_score

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
_MODEL_PATH = str(Path(__file__).parent / "face_landmarker.task")


def main():
    print("Starting webcam face detection test...")
    print("Press 'q' in the video window to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Loading FER detector...")
    fer = FERDetector(mtcnn=False)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Ready.\n")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe for face landmarks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_image)

            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    for lm in face_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

            # FER for emotion classification
            fer_results = fer.detect_emotions(frame)
            emotions = None
            dominant = None
            face_score = 0.0

            if fer_results:
                emotions = fer_results[0]["emotions"]
                dominant = max(emotions, key=emotions.get)
                face_score = _fer_engagement_score(emotions)

            # Draw status
            presence = 1.0 if results.face_landmarks else 0.0
            status = f"presence={presence:.0f}  face={face_score:.3f}"
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if dominant:
                cv2.putText(frame, f"emotion: {dominant}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            if emotions:
                y_offset = 90
                for emo, val in sorted(emotions.items(), key=lambda x: -x[1]):
                    bar_len = int(val * 200)
                    pct = f"{val*100:.1f}%"
                    color = (0, 255, 0) if emo in ("happy", "surprise") else (0, 150, 255)
                    cv2.putText(frame, f"{emo[:3]}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.rectangle(frame, (50, y_offset - 10), (50 + bar_len, y_offset), color, -1)
                    cv2.putText(frame, pct, (55 + bar_len, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 22

                emo_str = "  ".join(f"{k}={v*100:.0f}%" for k, v in sorted(emotions.items(), key=lambda x: -x[1]))
                print(f"\r{status}  [{dominant}] {emo_str}      ", end="", flush=True)

            cv2.imshow("ClaudeDJ Face Detection Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    print("\n\nStopping...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
