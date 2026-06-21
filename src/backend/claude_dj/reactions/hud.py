from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .reaction import ReactionFrame


def build_hud_snapshot(reaction_frame: ReactionFrame | None) -> dict[str, Any]:
    valence = _valence(reaction_frame)
    emotion_bucket = _emotion_bucket(valence)
    face_detected = bool(reaction_frame and float(reaction_frame.presence or 0.0) > 0.0)
    motion_energy = _motion_energy(reaction_frame.movement if reaction_frame else None)
    agent_action = None
    agent_reason = ""
    if emotion_bucket == "negative":
        agent_action = "change_track"
        agent_reason = "Listener disengaged (low valence + motion)."
    elif emotion_bucket == "positive":
        agent_action = "keep"
        agent_reason = "Listener looks engaged."
    return {
        "current_track": "(no track)",
        "bpm": 120.0,
        "emotion_bucket": emotion_bucket,
        "valence": round(valence, 2),
        "vibe_score": _clamp((valence + 1.0) / 2.0) if face_detected else 0.0,
        "motion_energy": motion_energy,
        "agent_action": agent_action,
        "agent_reason": agent_reason,
        "face_detected": face_detected,
    }


def draw_vibedj_hud(frame: Any, reaction_frame: ReactionFrame | None, pitch_history: Sequence[float]) -> Any:
    import cv2
    import numpy as np

    snap = build_hud_snapshot(reaction_frame)
    return _draw_hud(frame, snap, pitch_history, cv2, np)


def _valence(reaction_frame: ReactionFrame | None) -> float:
    if reaction_frame is None:
        return 0.0
    if reaction_frame.emotions:
        happy = float(reaction_frame.emotions.get("happy", 0.0))
        disinterested = float(reaction_frame.emotions.get("disinterested", 0.0))
        return _clamp_signed(happy - disinterested)
    if reaction_frame.raw_emotions:
        positive = float(reaction_frame.raw_emotions.get("happy", 0.0)) + float(
            reaction_frame.raw_emotions.get("surprise", 0.0)
        )
        negative = sum(
            float(reaction_frame.raw_emotions.get(emotion, 0.0))
            for emotion in ("sad", "angry", "fear", "disgust")
        )
        total = positive + negative + float(reaction_frame.raw_emotions.get("neutral", 0.0))
        if total > 0:
            return _clamp_signed((positive - negative) / total)
    if reaction_frame.face is not None:
        return _clamp_signed((float(reaction_frame.face) * 2.0) - 1.0)
    return 0.0


def _emotion_bucket(valence: float) -> str:
    if valence > 0.15:
        return "positive"
    if valence < -0.15:
        return "negative"
    return "neutral"


def _motion_energy(movement: float | None) -> float:
    if movement is None:
        return 0.0
    return round(_clamp(float(movement) / 15.0), 3)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _rounded_rect(cv2: Any, img: Any, pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[int, int, int], radius: int, thickness: int = -1) -> None:
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


def _pill_bar(
    cv2: Any,
    img: Any,
    x: int,
    y: int,
    w: int,
    h: int,
    ratio: float,
    fill_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
) -> None:
    radius = h // 2
    _rounded_rect(cv2, img, (x, y), (x + w, y + h), bg_color, radius, -1)
    if ratio > 0.01:
        fill_width = max(h, int(ratio * w))
        _rounded_rect(cv2, img, (x, y), (x + fill_width, y + h), fill_color, radius, -1)


def _draw_hud(frame: Any, snap: dict[str, Any], pitch_history: Sequence[float], cv2: Any, np: Any) -> Any:
    h, w = frame.shape[:2]

    pbg = (35, 28, 32)
    border = (200, 155, 210)
    border_glow = (140, 110, 150)
    text_white = (245, 240, 248)
    text_muted = (185, 178, 195)
    mint = (175, 235, 165)
    coral = (140, 145, 255)
    peach = (150, 210, 255)
    cyan = (225, 215, 145)
    pink = (205, 155, 250)
    bar_bg = (65, 55, 60)
    trace_bg = (30, 22, 28)
    trace_line = (210, 190, 255)
    separator = (80, 65, 75)

    pad = 12
    panel_width = 310
    px, py = pad, pad
    has_action = bool(snap.get("agent_action"))
    has_reason = has_action and bool(snap.get("agent_reason"))
    panel_height = 195 if has_reason else 180 if has_action else 148

    overlay = frame.copy()
    _rounded_rect(cv2, overlay, (px, py), (px + panel_width, py + panel_height), pbg, 14, -1)

    trace_height = 55
    tx, ty = pad, h - pad - trace_height
    trace_width = w - 2 * pad
    _rounded_rect(cv2, overlay, (tx, ty), (tx + trace_width, ty + trace_height), trace_bg, 10, -1)

    fcx, fcy = w - 28, 28
    cv2.circle(overlay, (fcx, fcy), 14, pbg, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    _rounded_rect(cv2, frame, (px - 1, py - 1), (px + panel_width + 1, py + panel_height + 1), border_glow, 15, 1)
    _rounded_rect(cv2, frame, (px, py), (px + panel_width, py + panel_height), border, 14, 1)
    _rounded_rect(cv2, frame, (tx, ty), (tx + trace_width, ty + trace_height), border, 10, 1)

    cx = px + pad
    cy = py + 22
    cv2.circle(frame, (cx + 3, cy - 4), 3, pink, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + 12, cy - 7), 2, mint, -1, cv2.LINE_AA)
    cv2.putText(frame, "VibeDJ", (cx + 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pink, 1, cv2.LINE_AA)

    sep_y = cy + 8
    cv2.line(frame, (cx, sep_y), (px + panel_width - pad, sep_y), separator, 1)

    cy = sep_y + 18
    track = str(snap.get("current_track") or "(no track)")
    if len(track) > 25:
        track = track[:24] + ".."
    cv2.putText(frame, track, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_white, 1, cv2.LINE_AA)
    bpm_text = f"{float(snap.get('bpm') or 120.0):.0f} bpm"
    bpm_size = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    cv2.putText(
        frame,
        bpm_text,
        (px + panel_width - pad - bpm_size[0], cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        text_muted,
        1,
        cv2.LINE_AA,
    )

    cy += 24
    bucket = str(snap.get("emotion_bucket") or "neutral")
    valence = float(snap.get("valence") or 0.0)
    emotion_color = {"positive": mint, "neutral": cyan, "negative": coral}.get(bucket, text_white)
    cv2.circle(frame, (cx + 4, cy - 4), 4, emotion_color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{bucket} ({valence:.2f})",
        (cx + 14, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        emotion_color,
        1,
        cv2.LINE_AA,
    )

    cy += 22
    vibe = float(snap.get("vibe_score") or 0.0)
    vibe_color = mint if vibe > 0.7 else cyan if vibe > 0.3 else (90, 80, 85)
    cv2.putText(frame, "vibe", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_muted, 1, cv2.LINE_AA)
    bx, bw, bh = cx + 42, 180, 12
    _pill_bar(cv2, frame, bx, cy - 10, bw, bh, vibe, vibe_color, bar_bg)
    cv2.putText(frame, f"{vibe:.0%}", (bx + bw + 6, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_white, 1, cv2.LINE_AA)

    cy += 20
    energy = float(snap.get("motion_energy") or 0.0)
    cv2.putText(frame, "move", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_muted, 1, cv2.LINE_AA)
    _pill_bar(cv2, frame, bx, cy - 10, bw, bh, energy, peach, bar_bg)
    cv2.putText(frame, f"{energy:.0%}", (bx + bw + 6, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_white, 1, cv2.LINE_AA)

    if has_action:
        cy += 22
        action = str(snap.get("agent_action"))
        reason = str(snap.get("agent_reason") or "")
        action_color = {"change_track": coral, "increase_energy": peach, "decrease_energy": cyan, "keep": mint}.get(action, pink)
        dx, dy = cx + 4, cy - 3
        diamond = np.array([[dx, dy - 4], [dx + 4, dy], [dx, dy + 4], [dx - 4, dy]], np.int32)
        cv2.fillPoly(frame, [diamond], action_color)
        cv2.putText(
            frame,
            action.replace("_", " "),
            (cx + 14, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            action_color,
            1,
            cv2.LINE_AA,
        )
        if reason:
            cy += 15
            reason_text = reason[:36] + (".." if len(reason) > 36 else "")
            cv2.putText(frame, reason_text, (cx + 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_muted, 1, cv2.LINE_AA)

    tip = 8
    t_x1, t_x2 = tx + tip, tx + trace_width - tip
    t_y1, t_y2 = ty + tip, ty + trace_height - tip
    t_dw, t_dh = t_x2 - t_x1, t_y2 - t_y1
    cv2.putText(frame, "pitch", (t_x1, t_y1 + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.25, text_muted, 1, cv2.LINE_AA)
    zero_y = t_y1 + t_dh // 2
    cv2.line(frame, (t_x1, zero_y), (t_x2, zero_y), (70, 60, 65), 1)

    if len(pitch_history) > 1:
        points = list(pitch_history)
        n = len(points)
        for index in range(1, n):
            x1 = t_x1 + int((index - 1) / max(n - 1, 1) * t_dw)
            x2 = t_x1 + int(index / max(n - 1, 1) * t_dw)
            y1 = t_y2 - int((points[index - 1] + 30) / 60 * t_dh)
            y2 = t_y2 - int((points[index] + 30) / 60 * t_dh)
            y1 = max(t_y1, min(t_y2, y1))
            y2 = max(t_y1, min(t_y2, y2))
            cv2.line(frame, (x1, y1), (x2, y2), trace_line, 1, cv2.LINE_AA)

    ring_color = mint if bool(snap.get("face_detected")) else coral
    cv2.circle(frame, (fcx, fcy), 10, ring_color, 2, cv2.LINE_AA)
    cv2.circle(frame, (fcx, fcy), 4, ring_color, -1, cv2.LINE_AA)
    return frame
