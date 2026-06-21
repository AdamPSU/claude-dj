"""Live webcam preview with reaction overlay for ClaudeDJ.

Shows the webcam feed with real-time landmark expressions, head pose,
emotion scores, and engagement signals overlaid. Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

from reaction import (
    COLLAPSED_KEYS, HeadPose, LandmarkExpression,
    RAW_TO_COLLAPSED, emotion_confidence,
)
from webcam import (
    _compute_landmark_expression, _estimate_head_pose,
    _head_movement, _preprocess_frame, _deepface_to_emotions,
    _smooth_emotions, _engagement_score,
    _MODEL_PATH,
)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_bar(frame, x, y, w, h, value, color, label="", max_val=1.0):
    """Draw a labeled horizontal bar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill_w = int(w * min(value / max_val, 1.0))
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 120), 1)
    if label:
        cv2.putText(frame, label, (x + 4, y + h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    val_text = f"{value:.2f}"
    cv2.putText(frame, val_text, (x + w + 5, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Warm up DeepFace
    print("Warming up DeepFace...")
    dummy = np.zeros((48, 48, 3), dtype=np.uint8)
    DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False,
                     silent=True, detector_backend="skip")
    print("Ready!")

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prev_pose = None
    smoothed_emo = None
    frame_count = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror
            ih, iw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_image)

            face_detected = len(results.face_landmarks) > 0

            # Dark overlay panel on the left
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (280, ih), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            panel_x = 10
            y_cursor = 25

            # Title
            cv2.putText(frame, "ClaudeDJ Reaction Preview", (panel_x, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
            y_cursor += 30

            if not face_detected:
                cv2.putText(frame, "No face detected", (panel_x, y_cursor),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("ClaudeDJ Preview", frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                continue

            lmarks = results.face_landmarks[0]

            # Draw face mesh points
            for lm in lmarks:
                px, py = int(lm.x * iw), int(lm.y * ih)
                cv2.circle(frame, (px, py), 1, (0, 180, 0), -1)

            # Head pose
            head_pose = _estimate_head_pose(lmarks, iw, ih)
            movement = _head_movement(prev_pose, head_pose)
            prev_pose = head_pose

            # Landmark expressions
            lm_expr = _compute_landmark_expression(lmarks, iw, ih)

            # Face area
            xs = [lm.x * iw for lm in lmarks]
            ys = [lm.y * ih for lm in lmarks]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            face_area = float((x2 - x1) * (y2 - y1)) / (iw * ih)

            # Draw face bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # DeepFace (every 3rd frame to stay responsive)
            raw_emo = None
            collapsed_emo = None
            face_conf = 0.0
            face_score = 0.0
            dominant = "?"

            if frame_count % 3 == 0:
                margin = int(0.2 * (x2 - x1))
                face_crop = frame[
                    max(0, y1 - margin):min(ih, y2 + margin),
                    max(0, x1 - margin):min(iw, x2 + margin),
                ]
                if face_crop.size > 0:
                    enhanced = _preprocess_frame(face_crop)
                    try:
                        df_results = DeepFace.analyze(
                            enhanced, actions=["emotion"],
                            enforce_detection=False, silent=True,
                            detector_backend="skip",
                        )
                        if df_results:
                            result = df_results[0] if isinstance(df_results, list) else df_results
                            raw_emo, collapsed = _deepface_to_emotions(result["emotion"])
                            face_conf = emotion_confidence(collapsed)
                            collapsed_emo = _smooth_emotions(collapsed, smoothed_emo, confidence=face_conf)
                            smoothed_emo = collapsed_emo
                            face_score = _engagement_score(
                                collapsed_emo,
                                landmark_expr=lm_expr,
                                movement=movement,
                                head_pose=head_pose,
                            )
                            dominant = max(raw_emo, key=raw_emo.get)
                    except Exception:
                        pass

            if smoothed_emo is None and collapsed_emo is not None:
                smoothed_emo = collapsed_emo

            # --- Draw overlay ---

            # Section: Head Pose
            cv2.putText(frame, "HEAD POSE", (panel_x, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            y_cursor += 20

            if head_pose:
                yaw_color = (0, 255, 0) if abs(head_pose.yaw) < 20 else (0, 0, 255)
                cv2.putText(frame, f"Yaw: {head_pose.yaw:+.1f}", (panel_x, y_cursor),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, yaw_color, 1)
                cv2.putText(frame, f"Pitch: {head_pose.pitch:+.1f}", (panel_x + 90, y_cursor),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                cv2.putText(frame, f"Roll: {head_pose.roll:+.1f}", (panel_x + 185, y_cursor),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                y_cursor += 18
                cv2.putText(frame, f"Movement: {movement:.3f}", (panel_x, y_cursor),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

                # Looking away indicator
                if abs(head_pose.yaw) > 20:
                    cv2.putText(frame, "LOOKING AWAY", (panel_x + 140, y_cursor),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            y_cursor += 28

            # Section: Landmark Expressions
            cv2.putText(frame, "LANDMARKS", (panel_x, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            y_cursor += 20

            bar_w = 140
            bar_h = 14

            draw_bar(frame, panel_x, y_cursor, bar_w, bar_h,
                     lm_expr.smile, (0, 200, 100), "Smile")
            y_cursor += bar_h + 4

            draw_bar(frame, panel_x, y_cursor, bar_w, bar_h,
                     lm_expr.mouth_open, (200, 150, 0), "Mouth")
            y_cursor += bar_h + 4

            draw_bar(frame, panel_x, y_cursor, bar_w, bar_h,
                     lm_expr.ear, (150, 100, 200), "EAR", max_val=0.4)
            y_cursor += bar_h + 4

            draw_bar(frame, panel_x, y_cursor, bar_w, bar_h,
                     lm_expr.brow_height, (100, 180, 200), "Brow")
            y_cursor += bar_h + 8

            # Section: Face Area (lean-in)
            draw_bar(frame, panel_x, y_cursor, bar_w, bar_h,
                     face_area, (200, 200, 0), "FaceArea", max_val=0.3)
            y_cursor += bar_h + 12

            # Section: Emotions (DeepFace)
            cv2.putText(frame, "EMOTIONS (DeepFace)", (panel_x, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            y_cursor += 18

            cv2.putText(frame, f"Dominant: {dominant}", (panel_x, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 100), 1)
            cv2.putText(frame, f"Conf: {face_conf:.2f}", (panel_x + 150, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            y_cursor += 20

            if smoothed_emo:
                emo_colors = {
                    "happy": (0, 255, 0),
                    "neutral": (200, 200, 0),
                    "disinterested": (0, 0, 255),
                }
                for key in COLLAPSED_KEYS:
                    val = smoothed_emo.get(key, 0.0)
                    draw_bar(frame, panel_x, y_cursor, bar_w, bar_h,
                             val, emo_colors.get(key, (150, 150, 150)), key)
                    y_cursor += bar_h + 4

            y_cursor += 8

            # Section: Engagement Score
            cv2.putText(frame, "ENGAGEMENT", (panel_x, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            y_cursor += 20

            score_color = (0, 255, 0) if face_score > 0.6 else (0, 200, 255) if face_score > 0.4 else (0, 0, 255)
            draw_bar(frame, panel_x, y_cursor, bar_w + 60, 20,
                     face_score, score_color, "Score")
            y_cursor += 30

            # Engagement label
            if face_score > 0.6:
                label = "LIKES IT"
                label_color = (0, 255, 0)
            elif face_score > 0.4:
                label = "NEUTRAL"
                label_color = (0, 200, 255)
            else:
                label = "NOT FEELING IT"
                label_color = (0, 0, 255)

            cv2.putText(frame, label, (panel_x, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

            frame_count += 1
            cv2.imshow("ClaudeDJ Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
