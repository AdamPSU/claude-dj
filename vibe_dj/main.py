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


def _rounded_rect(img, pt1, pt2, color, radius, thickness=-1):
    """Draw a rounded rectangle (filled or outline)."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x1 + r, y2 - r), color, -1)
        cv2.rectangle(img, (x2 - r, y1 + r), (x2, y2 - r), color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.line(img, (x2 - r, y2), (x1 + r, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y1 + r), color, thickness)


def _pill_bar(img, x, y, w, h, ratio, fill_color, bg_color):
    """Draw a pill-shaped progress bar."""
    r = h // 2
    _rounded_rect(img, (x, y), (x + w, y + h), bg_color, r, -1)
    if ratio > 0.01:
        fw = max(h, int(ratio * w))
        _rounded_rect(img, (x, y), (x + fw, y + h), fill_color, r, -1)


def draw_hud(frame: np.ndarray, snap: dict, pitch_history: deque) -> np.ndarray:
    """Draw a clean HUD overlay with rounded panels and pastel accents."""
    h, w = frame.shape[:2]

    # Palette (BGR)
    PBG = (35, 28, 32)
    BDR = (200, 155, 210)
    BDR_GLOW = (140, 110, 150)
    TW = (245, 240, 248)
    TM = (185, 178, 195)
    MINT = (175, 235, 165)
    CORAL = (140, 145, 255)
    PEACH = (150, 210, 255)
    CYAN = (225, 215, 145)
    PINK = (205, 155, 250)
    BBG = (65, 55, 60)
    TBG = (30, 22, 28)
    TLINE = (210, 190, 255)
    SEP = (80, 65, 75)

    pad = 12
    pw = 310
    px, py = pad, pad

    has_action = bool(snap.get("agent_action"))
    has_reason = has_action and bool(snap.get("agent_reason"))
    ph = 148
    if has_action:
        ph = 195 if has_reason else 180

    # --- Pass 1: semi-transparent panel backgrounds ---
    overlay = frame.copy()
    _rounded_rect(overlay, (px, py), (px + pw, py + ph), PBG, 14, -1)

    trace_th = 55
    tx, ty = pad, h - pad - trace_th
    tw = w - 2 * pad
    _rounded_rect(overlay, (tx, ty), (tx + tw, ty + trace_th), TBG, 10, -1)

    fcx, fcy = w - 28, 28
    cv2.circle(overlay, (fcx, fcy), 14, PBG, -1)

    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # --- Pass 2: crisp borders + content ---

    # Panel border (outer glow + inner)
    _rounded_rect(frame, (px - 1, py - 1),
                  (px + pw + 1, py + ph + 1), BDR_GLOW, 15, 1)
    _rounded_rect(frame, (px, py), (px + pw, py + ph), BDR, 14, 1)

    # Trace border
    _rounded_rect(frame, (tx, ty), (tx + tw, ty + trace_th), BDR, 10, 1)

    cx = px + pad
    cy = py + 22

    # Title with decorative dots
    cv2.circle(frame, (cx + 3, cy - 4), 3, PINK, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + 12, cy - 7), 2, MINT, -1, cv2.LINE_AA)
    cv2.putText(frame, "VibeDJ", (cx + 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, PINK, 1, cv2.LINE_AA)

    sep_y = cy + 8
    cv2.line(frame, (cx, sep_y), (px + pw - pad, sep_y), SEP, 1)

    # Track + BPM
    cy = sep_y + 18
    track = snap["current_track"] or "(no track)"
    if len(track) > 25:
        track = track[:24] + ".."
    cv2.putText(frame, track, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TW, 1, cv2.LINE_AA)
    bpm_t = f"{snap['bpm']:.0f} bpm"
    bsz = cv2.getTextSize(bpm_t, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    cv2.putText(frame, bpm_t, (px + pw - pad - bsz[0], cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TM, 1, cv2.LINE_AA)

    # Emotion
    cy += 24
    bucket = snap["emotion_bucket"]
    valence = snap["valence"]
    ecol = {"positive": MINT, "neutral": CYAN, "negative": CORAL}.get(bucket, TW)
    cv2.circle(frame, (cx + 4, cy - 4), 4, ecol, -1, cv2.LINE_AA)
    cv2.putText(frame, f"{bucket} ({valence:.2f})", (cx + 14, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, ecol, 1, cv2.LINE_AA)

    # Vibe pill bar
    cy += 22
    vibe = snap["vibe_score"]
    vcol = MINT if vibe > 0.7 else CYAN if vibe > 0.3 else (90, 80, 85)
    cv2.putText(frame, "vibe", (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TM, 1, cv2.LINE_AA)
    bx, bw, bh = cx + 42, 180, 12
    _pill_bar(frame, bx, cy - 10, bw, bh, vibe, vcol, BBG)
    cv2.putText(frame, f"{vibe:.0%}", (bx + bw + 6, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TW, 1, cv2.LINE_AA)

    # Motion pill bar
    cy += 20
    energy = snap["motion_energy"]
    cv2.putText(frame, "move", (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TM, 1, cv2.LINE_AA)
    _pill_bar(frame, bx, cy - 10, bw, bh, energy, PEACH, BBG)
    cv2.putText(frame, f"{energy:.0%}", (bx + bw + 6, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TW, 1, cv2.LINE_AA)

    # Agent decision
    if has_action:
        cy += 22
        action = snap["agent_action"]
        reason = snap.get("agent_reason", "")
        dcol = {"change_track": CORAL, "increase_energy": PEACH,
                "decrease_energy": CYAN, "keep": MINT}.get(action, PINK)
        dx, dy = cx + 4, cy - 3
        dia = np.array([[dx, dy - 4], [dx + 4, dy],
                        [dx, dy + 4], [dx - 4, dy]], np.int32)
        cv2.fillPoly(frame, [dia], dcol)
        cv2.putText(frame, action.replace("_", " "), (cx + 14, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, dcol, 1, cv2.LINE_AA)
        if reason:
            cy += 15
            rtxt = reason[:36] + (".." if len(reason) > 36 else "")
            cv2.putText(frame, rtxt, (cx + 14, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, TM, 1, cv2.LINE_AA)

    # --- Pitch trace ---
    tip = 8
    t_x1, t_x2 = tx + tip, tx + tw - tip
    t_y1, t_y2 = ty + tip, ty + trace_th - tip
    t_dw, t_dh = t_x2 - t_x1, t_y2 - t_y1

    cv2.putText(frame, "pitch", (t_x1, t_y1 + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, TM, 1, cv2.LINE_AA)

    zero_y = t_y1 + t_dh // 2
    cv2.line(frame, (t_x1, zero_y), (t_x2, zero_y), (70, 60, 65), 1)

    if len(pitch_history) > 1:
        pts = list(pitch_history)
        n = len(pts)
        for i in range(1, n):
            x1 = t_x1 + int((i - 1) / max(n - 1, 1) * t_dw)
            x2 = t_x1 + int(i / max(n - 1, 1) * t_dw)
            y1 = t_y2 - int((pts[i - 1] + 30) / 60 * t_dh)
            y2 = t_y2 - int((pts[i] + 30) / 60 * t_dh)
            y1 = max(t_y1, min(t_y2, y1))
            y2 = max(t_y1, min(t_y2, y2))
            cv2.line(frame, (x1, y1), (x2, y2), TLINE, 1, cv2.LINE_AA)

    # Face detection indicator (ring + dot)
    rc = MINT if snap["face_detected"] else CORAL
    cv2.circle(frame, (fcx, fcy), 10, rc, 2, cv2.LINE_AA)
    cv2.circle(frame, (fcx, fcy), 4, rc, -1, cv2.LINE_AA)

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
