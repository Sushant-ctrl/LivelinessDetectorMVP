"""OpenCV drawing utilities for the liveness detector UI."""

import cv2
import numpy as np

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_GREEN  = (50,  220,  50)
_RED    = (50,   50, 220)
_YELLOW = (0,   210, 210)
_ORANGE = (0,   155, 255)
_WHITE  = (235, 235, 235)
_DARK   = (15,   15,  15)
_BLUE   = (200, 130,  60)

PANEL_H = 110


def render(frame: np.ndarray, face_data, state: str, runner,
           avg_texture: float, verdict: "str | None", low_texture: bool,
           hand_data=None, dev_label: "str | None" = None):
    _draw_header(frame, dev_label)
    _draw_face_box(frame, face_data, state)
    _draw_metrics(frame, face_data)
    if state == "CHALLENGE":
        _draw_challenge_banner(frame, runner, hand_data)
    if low_texture and state == "CHALLENGE":
        h = frame.shape[0]
        _put(frame, "Low texture | possible spoof detected",
             (12, h - PANEL_H - 10), 0.50, _ORANGE)
    _draw_panel(frame, state, runner, avg_texture, verdict, hand_data)


# ── internal helpers ──────────────────────────────────────────────────────────

def _draw_header(frame: np.ndarray, dev_label: "str | None" = None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 36), _DARK, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    _put(frame, "LIVENESS DETECTOR", (12, 25), 0.65, _WHITE, thickness=1)
    _put(frame, "Q = quit   R = restart", (frame.shape[1] - 220, 25), 0.45, _WHITE, thickness=1)
    if dev_label is not None:
        badge = f"DEV | {dev_label}"
        (tw, _), _ = cv2.getTextSize(badge, _FONT, 0.45, 1)
        cx = frame.shape[1] // 2
        cv2.putText(frame, badge, (cx - tw // 2, 25), _FONT, 0.45, _GREEN, 1, cv2.LINE_AA)


def _draw_face_box(frame: np.ndarray, face_data, state: str):
    if face_data is None:
        return
    x1, y1, x2, y2 = face_data.face_bbox
    color = _GREEN if state in ("PASSIVE", "CHALLENGE", "PASSED") else _RED
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    _corner_marks(frame, x1, y1, x2, y2, color, length=18, thickness=3)


def _corner_marks(frame, x1, y1, x2, y2, color, length=18, thickness=3):
    for (px, py), (dx, dy) in [
        ((x1, y1), (+1, 0)), ((x1, y1), (0, +1)),
        ((x2, y1), (-1, 0)), ((x2, y1), (0, +1)),
        ((x1, y2), (+1, 0)), ((x1, y2), (0, -1)),
        ((x2, y2), (-1, 0)), ((x2, y2), (0, -1)),
    ]:
        cv2.line(frame, (px, py), (px + dx * length, py + dy * length),
                 color, thickness, cv2.LINE_AA)


def _draw_metrics(frame: np.ndarray, face_data):
    if face_data is None:
        return
    items = [
        f"EAR  {face_data.ear:.3f}",
        f"Yaw  {face_data.head_yaw:+.1f}d",
        f"Tex  {face_data.texture_score:.0f}",
    ]
    for i, text in enumerate(items):
        cv2.putText(frame, text, (12, 54 + i * 20),
                    _FONT, 0.42, _WHITE, 1, cv2.LINE_AA)


def _draw_challenge_banner(frame: np.ndarray, runner, hand_data=None):
    """Large top banner visible in peripheral vision during head-turn challenges."""
    from challenges import HAND_CHALLENGES
    c = runner.current
    if c is None:
        return

    h, w = frame.shape[:2]
    BANNER_TOP    = 36   # sits just below the header
    BANNER_HEIGHT = 82

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, BANNER_TOP), (w, BANNER_TOP + BANNER_HEIGHT), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, BANNER_TOP + BANNER_HEIGHT), (w, BANNER_TOP + BANNER_HEIGHT), _WHITE, 1)

    t     = c.time_remaining
    color = _GREEN if t > 2.0 else _RED

    # Large instruction text — centred
    text_y = BANNER_TOP + 46
    _put_center(frame, c.instruction, w // 2, text_y, 1.4, _YELLOW, thickness=3)

    # Timer — top-right, large
    timer_str = f"{t:.1f}s"
    (tw, _), _ = cv2.getTextSize(timer_str, _FONT, 1.2, 2)
    cv2.putText(frame, timer_str, (w - tw - 16, text_y),
                _FONT, 1.2, color, 2, cv2.LINE_AA)

    # Thick full-width progress bar
    bar_y = BANNER_TOP + BANNER_HEIGHT - 22
    _progress_bar(frame, t / c.timeout, (0, bar_y), (w, 18), color)

    # Hint for hand challenges when no hand visible
    if c.ctype in HAND_CHALLENGES and hand_data is None:
        _put_center(frame, "Show your hand to the camera",
                    w // 2, BANNER_TOP + BANNER_HEIGHT - 28, 0.52, _WHITE, thickness=1)


def _draw_panel(frame: np.ndarray, state: str, runner,
                avg_texture: float, verdict: "str | None", hand_data=None):
    h, w = frame.shape[:2]
    y0 = h - PANEL_H
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), _DARK, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.line(frame, (0, y0), (w, y0), _WHITE, 1)

    cy  = y0 + 36   # main text y
    by  = y0 + 52   # progress bar y
    dot_y = h - 16  # dots row y

    if verdict == "LIVE":
        _put_center(frame, "VERIFICATION PASSED  |  LIVE", w // 2, cy, 0.9, _GREEN)
        _put_center(frame, "Press  R  to run again", w // 2, cy + 38, 0.52, _WHITE)

    elif verdict == "FAILED":
        _put_center(frame, "VERIFICATION FAILED", w // 2, cy, 0.9, _RED)
        _put_center(frame, "Press  R  to retry", w // 2, cy + 38, 0.52, _WHITE)

    elif state == "NO_FACE":
        _put_center(frame, "Position your face in the frame", w // 2, cy + 18, 0.72, _YELLOW)

    elif state == "PASSIVE":
        pct = min(1.0, avg_texture / 80.0)
        _put(frame, f"Analyzing face texture ...  {int(pct * 100)}%", (16, cy), 0.70, _WHITE)
        _progress_bar(frame, pct, (16, by), (w - 32, 18), _BLUE)
        _draw_dots(frame, runner, w, dot_y)

    elif state == "CHALLENGE":
        from challenges import HAND_CHALLENGES
        c = runner.current
        if c is not None:
            _put(frame, c.instruction, (16, cy), 0.82, _YELLOW)
            t = c.time_remaining
            color = _GREEN if t > 2.0 else _RED
            _put(frame, f"{t:.1f}s", (w - 72, cy), 0.80, color)
            _progress_bar(frame, t / c.timeout, (16, by), (w - 32, 18), color)
            if c.ctype in HAND_CHALLENGES and hand_data is None:
                _put(frame, "Show your hand to the camera", (16, by + 28), 0.52, _WHITE)
        _draw_dots(frame, runner, w, dot_y)


def _draw_dots(frame, runner, w: int, y: int):
    total   = len(runner.challenges)
    r       = 8
    spacing = 30
    sx      = w // 2 - (total * spacing) // 2
    for i in range(total):
        cx = sx + i * spacing + spacing // 2
        if i < runner.index:
            cv2.circle(frame, (cx, y), r, _GREEN, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, y), r, _WHITE,  1, cv2.LINE_AA)
        elif i == runner.index:
            cv2.circle(frame, (cx, y), r, _YELLOW, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, y), r, _WHITE,  1, cv2.LINE_AA)


def _progress_bar(frame, progress: float, pos: tuple, size: tuple, color: tuple):
    x, y   = pos
    bw, bh = size
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), _WHITE, 1)
    fw = int(bw * max(0.0, min(1.0, progress)))
    if fw > 2:
        cv2.rectangle(frame, (x + 1, y + 1), (x + fw - 1, y + bh - 1), color, -1)


def _put(frame, text: str, pos: tuple, scale: float, color: tuple, thickness: int = 2):
    cv2.putText(frame, text, pos, _FONT, scale, color, thickness, cv2.LINE_AA)


def _put_center(frame, text: str, cx: int, y: int, scale: float, color: tuple, thickness: int = 2):
    (tw, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    cv2.putText(frame, text, (cx - tw // 2, y), _FONT, scale, color, thickness, cv2.LINE_AA)
