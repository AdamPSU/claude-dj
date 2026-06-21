"""Orchestration loop: video thread + audio thread + main fusion + HUD.

Usage:
    python -m vibe_dj.main --tracks tracks/

Press 'q' in the HUD window to quit.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np

from vibe_dj import config
from vibe_dj.state import SystemState
from vibe_dj.face import FaceProcessor
from vibe_dj.emotion import DeepFaceClassifier, ema_smooth, to_valence
from vibe_dj.beats import LibrosaGrid
from vibe_dj.vibe import VibeDetector
from vibe_dj.agent import DJAgent
from vibe_dj.player import Player


# ── Video thread ─────────────────────────────────────────────────────


def video_loop(state: SystemState, stop_event: threading.Event) -> None:
    """Capture frames, extract head pose, and classify emotion."""
    face_proc = FaceProcessor()

    print("[video] Warming up DeepFace (first call downloads weights)...")
    classifier = DeepFaceClassifier()
    print("[video] DeepFace ready.")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[video] ERROR: cannot open webcam")
        return

    prev_pitch: float | None = None
    prev_smoothed: dict[str, float] | None = None
    frame_count = 0
    interval = 1.0 / config.FPS_TARGET

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(interval)
            continue

        result = face_proc.process(frame)
        if result is not None:
            now = time.time()

            # Update head pose in state
            state.update_pose(
                result.pitch, result.yaw, result.roll,
                result.face_scale, True,
            )

            # Append RAW pitch to buffer (unsmoothed -- for vibe DSP)
            state.append_pitch(now, result.pitch)

            # Motion energy: magnitude of pitch change
            if prev_pitch is not None:
                delta = abs(result.pitch - prev_pitch)
                energy = min(1.0, delta / 15.0)  # ~15 deg = full energy
                with state._lock:
                    state.motion_energy = round(energy, 3)
            prev_pitch = result.pitch

            # Set face crop for emotion (copy-on-write)
            state.set_face_crop(result.face_crop)

            # Run emotion at lower cadence (every Nth frame)
            frame_count += 1
            if frame_count % config.EMOTION_CADENCE == 0:
                crop = state.get_face_crop()
                if crop is not None and crop.size > 0:
                    raw_probs = classifier.classify(crop)
                    smoothed = ema_smooth(raw_probs, prev_smoothed)
                    prev_smoothed = smoothed
                    valence = to_valence(smoothed)
                    bucket = max(smoothed, key=smoothed.get)
                    state.update_emotion(valence, bucket, smoothed)
        else:
            state.update_pose(0.0, 0.0, 0.0, 1.0, False)
            prev_pitch = None

        # Store latest frame for HUD
        with state._lock:
            state._latest_frame = frame

        time.sleep(max(0, interval - 0.005))

    cap.release()
    face_proc.close()


# ── Audio thread ─────────────────────────────────────────────────────


def audio_loop(
    state: SystemState, player: Player,
    stop_event: threading.Event,
) -> None:
    """Monitor playback and update state with position."""
    while not stop_event.is_set():
        if player.is_playing:
            pos = player.get_position_s()
            state.update_playback(pos, player.current_track, True)
        else:
            state.update_playback(0.0, player.current_track, False)
        time.sleep(0.1)


# ── HUD drawing ──────────────────────────────────────────────────────


def draw_hud(frame: np.ndarray, snap: dict, pitch_history: deque) -> np.ndarray:
    """Draw the debug HUD overlay onto the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    y = 25
    # Track + BPM
    track = snap["current_track"] or "(no track)"
    cv2.putText(overlay, f"Track: {track}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 25
    cv2.putText(overlay, f"BPM: {snap['bpm']:.0f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Emotion
    y += 35
    bucket = snap["emotion_bucket"]
    valence = snap["valence"]
    emo_color = {
        "positive": (0, 255, 0),
        "neutral": (200, 200, 200),
        "negative": (0, 0, 255),
    }.get(bucket, (255, 255, 255))
    cv2.putText(overlay, f"Emotion: {bucket} ({valence:.2f})", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, emo_color, 2)

    # Vibe bar
    y += 35
    vibe = snap["vibe_score"]
    vibe_w = int(vibe * 200)
    vibe_color = ((0, 255, 0) if vibe > 0.7
                  else (0, 200, 255) if vibe > 0.3
                  else (100, 100, 100))
    cv2.putText(overlay, "Vibe:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(overlay, (70, y - 15), (70 + vibe_w, y), vibe_color, -1)
    cv2.rectangle(overlay, (70, y - 15), (270, y), (100, 100, 100), 1)
    cv2.putText(overlay, f"{vibe:.2f}", (280, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Motion energy bar
    y += 30
    energy = snap["motion_energy"]
    en_w = int(energy * 200)
    cv2.putText(overlay, "Move:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(overlay, (70, y - 15), (70 + en_w, y), (255, 200, 0), -1)
    cv2.rectangle(overlay, (70, y - 15), (270, y), (100, 100, 100), 1)
    cv2.putText(overlay, f"{energy:.2f}", (280, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Agent decision
    y += 35
    action = snap.get("agent_action", "")
    reason = snap.get("agent_reason", "")
    if action:
        dj_color = {
            "change_track": (0, 0, 255),
            "increase_energy": (0, 200, 255),
            "decrease_energy": (255, 200, 0),
            "keep": (0, 255, 0),
        }.get(action, (0, 255, 255))
        cv2.putText(overlay, f"DJ: {action}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, dj_color, 2)
        y += 22
        cv2.putText(overlay, reason, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # --- Pitch trace (bottom of frame) ---
    trace_y_base = h - 20
    trace_h = 60
    trace_x_start = 10
    trace_w_px = w - 20

    # Background for trace area
    cv2.rectangle(overlay, (trace_x_start, trace_y_base - trace_h),
                  (trace_x_start + trace_w_px, trace_y_base), (30, 30, 30), -1)
    cv2.putText(overlay, "pitch", (trace_x_start + 2, trace_y_base - trace_h + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    if len(pitch_history) > 1:
        pts = list(pitch_history)
        n = len(pts)
        for i in range(1, n):
            x1 = trace_x_start + int((i - 1) / max(n - 1, 1) * trace_w_px)
            x2 = trace_x_start + int(i / max(n - 1, 1) * trace_w_px)
            p1_val = pts[i - 1]
            p2_val = pts[i]
            y1 = trace_y_base - int((p1_val + 30) / 60 * trace_h)
            y2 = trace_y_base - int((p2_val + 30) / 60 * trace_h)
            y1 = max(trace_y_base - trace_h, min(trace_y_base, y1))
            y2 = max(trace_y_base - trace_h, min(trace_y_base, y2))
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Zero line
    zero_y = trace_y_base - trace_h // 2
    cv2.line(overlay, (trace_x_start, zero_y),
             (trace_x_start + trace_w_px, zero_y), (80, 80, 80), 1)

    # Face detection indicator
    face_color = (0, 255, 0) if snap["face_detected"] else (0, 0, 255)
    cv2.circle(overlay, (w - 20, 20), 8, face_color, -1)

    # Blend overlay
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeDJ -- real-time emotion + vibe DJ")
    parser.add_argument("--tracks", default=config.TRACKS_DIR,
                        help="Directory containing audio files")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    config.CAMERA_INDEX = args.camera
    config.TRACKS_DIR = args.tracks

    state = SystemState()
    state._latest_frame = None  # shared frame for HUD

    vibe_detector = VibeDetector()
    agent = DJAgent()
    player = Player(tracks_dir=args.tracks)
    beat_source = LibrosaGrid()

    # Load tracks
    track_names = player.load_tracks()
    has_audio = bool(track_names)
    if not has_audio:
        print(f"No audio files in '{args.tracks}/'.")
        print("Running in webcam-only mode (no audio).\n")

    stop_event = threading.Event()

    # Start video thread
    video_thread = threading.Thread(
        target=video_loop, args=(state, stop_event), daemon=True,
    )
    video_thread.start()

    # Start audio if tracks available
    if has_audio:
        print(f"[main] Loading beat grid for: {track_names[0]}")
        beat_source.load(player._tracks[0])
        state.update_beats(beat_source.bpm, beat_source.all_beat_times)
        print(f"[main] BPM: {beat_source.bpm:.1f}, {len(beat_source.all_beat_times)} beats")
        player.play(0)

        audio_thread = threading.Thread(
            target=audio_loop, args=(state, player, stop_event),
            daemon=True,
        )
        audio_thread.start()

    pitch_history: deque[float] = deque(maxlen=150)
    fusion_interval = 1.0 / config.FUSION_HZ

    print("[main] Running. Press 'q' in the HUD window to quit.")

    while True:
        # --- Fusion tick ---
        snap = state.snapshot()

        # Append latest pitch to HUD trace
        pitch_history.append(snap["pitch"])

        # Compute vibe from raw pitch window
        pitch_window = state.get_pitch_window(config.WINDOW_LEN_S)
        playback_t = snap["playback_time"]
        beats_in_window = beat_source.beats_in_window(
            max(0, playback_t - config.WINDOW_LEN_S), playback_t,
        ) if snap["is_playing"] else []
        vibe, plv, pm = vibe_detector.compute(
            pitch_window, beats_in_window, snap["bpm"],
        )
        state.update_vibe(vibe, plv, pm)
        snap["vibe_score"] = vibe

        # Agent decision
        decision = agent.decide(snap["valence"], vibe, snap["motion_energy"])
        state.update_agent(decision.action, decision.reason)
        snap["agent_action"] = decision.action
        snap["agent_reason"] = decision.reason

        # Act on agent decision
        if decision.action == "change_track" and has_audio and player.track_count > 1:
            new_track = player.next_track()
            print(f"\n[agent] Changing track -> {new_track} ({decision.reason})")
            if player.current_path:
                beat_source.load(player.current_path)
                state.update_beats(beat_source.bpm, beat_source.all_beat_times)

        # --- HUD ---
        with state._lock:
            frame = getattr(state, "_latest_frame", None)
            if frame is not None:
                frame = frame.copy()

        if frame is not None:
            hud_frame = draw_hud(frame, snap, pitch_history)
            cv2.imshow("VibeDJ", hud_frame)

        key = cv2.waitKey(max(1, int(fusion_interval * 1000))) & 0xFF
        if key == ord("q"):
            break

    # Cleanup
    print("\n[main] Shutting down...")
    stop_event.set()
    if has_audio:
        player.stop()
    video_thread.join(timeout=3.0)
    cv2.destroyAllWindows()
    print("[main] Done.")


if __name__ == "__main__":
    main()
