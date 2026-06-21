"""Manual verification: webcam + FaceMesh -> live pitch/yaw/roll printed."""

import cv2
from vibe_dj.face import FaceProcessor


def main():
    proc = FaceProcessor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam")
        return

    print("Press 'q' to quit. Showing live head pose...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = proc.process(frame)
        if result:
            txt = (f"pitch={result.pitch:+6.1f}  yaw={result.yaw:+6.1f}  "
                   f"roll={result.roll:+6.1f}  scale={result.face_scale:.2f}")
            print(f"\r{txt}   ", end="", flush=True)
            cv2.putText(frame, txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw landmarks
            h, w = frame.shape[:2]
            for lm in result.landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
        else:
            cv2.putText(frame, "No face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Face Pose Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    proc.close()
    print()


if __name__ == "__main__":
    main()
