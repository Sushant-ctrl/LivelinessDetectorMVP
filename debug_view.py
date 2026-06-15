"""Developer landmark debug window.

Activated via:  pythonw main.py --dev

Shows a separate window with every MediaPipe landmark point rendered in
real time — face tesselation, feature contours, and per-finger hand
connections — alongside the live metric readouts.
"""

import cv2
import numpy as np
from mediapipe.tasks.python import vision as _mp_vision
from challenges import ChallengeType

# ── face connection groups with (BGR colour, thickness) ──────────────────────
_FC = _mp_vision.FaceLandmarksConnections
_FACE_LAYERS = [
    (_FC.FACE_LANDMARKS_TESSELATION,    (40,  40,  40), 1),   # dark mesh
    (_FC.FACE_LANDMARKS_FACE_OVAL,      (160, 160, 160), 1),  # outer oval
    (_FC.FACE_LANDMARKS_LEFT_EYE,       (220, 200,  50), 2),  # cyan-ish
    (_FC.FACE_LANDMARKS_RIGHT_EYE,      (220, 200,  50), 2),
    (_FC.FACE_LANDMARKS_LEFT_EYEBROW,   (80,  200, 200), 1),
    (_FC.FACE_LANDMARKS_RIGHT_EYEBROW,  (80,  200, 200), 1),
    (_FC.FACE_LANDMARKS_LIPS,           (80,  100, 220), 2),  # reddish
    (_FC.FACE_LANDMARKS_NOSE,           (100, 220, 100), 1),  # green
    (_FC.FACE_LANDMARKS_LEFT_IRIS,      (220, 100,  50), 2),  # bright teal
    (_FC.FACE_LANDMARKS_RIGHT_IRIS,     (220, 100,  50), 2),
]

# ── hand connection groups with (BGR colour, thickness) ──────────────────────
_HC = _mp_vision.HandLandmarksConnections
_HAND_LAYERS = [
    (_HC.HAND_PALM_CONNECTIONS,         (200, 200, 200), 2),
    (_HC.HAND_THUMB_CONNECTIONS,        (50,   50, 240), 3),  # red
    (_HC.HAND_INDEX_FINGER_CONNECTIONS, (50,  150, 240), 2),  # orange
    (_HC.HAND_MIDDLE_FINGER_CONNECTIONS,(50,  230, 230), 2),  # yellow
    (_HC.HAND_RING_FINGER_CONNECTIONS,  (50,  200,  80), 2),  # green
    (_HC.HAND_PINKY_FINGER_CONNECTIONS, (230, 100,  50), 2),  # blue
]

# Key face landmarks to label by name
_LABELLED = {
    1:   "NOSE",
    33:  "EYE-L",
    263: "EYE-R",
    4:   "TIP",
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_connections(canvas, lm, connections, colour, thickness, w, h):
    for conn in connections:
        p1 = lm[conn.start]
        p2 = lm[conn.end]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(canvas, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)


def _draw_all_face_points(canvas, lm, w, h):
    """Draw every landmark as a small dot; highlight liveness-relevant ones."""
    for i, pt in enumerate(lm):
        x, y = int(pt.x * w), int(pt.y * h)
        if i in _LABELLED:
            cv2.circle(canvas, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, _LABELLED[i], (x + 6, y - 4),
                        _FONT, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (x, y), 1, (180, 180, 180), -1, cv2.LINE_AA)


def _draw_hand_points(canvas, lm, w, h):
    finger_colours = [
        (50, 50, 240),   # thumb   (idx 1-4)
        (50, 150, 240),  # index   (idx 5-8)
        (50, 230, 230),  # middle  (idx 9-12)
        (50, 200, 80),   # ring    (idx 13-16)
        (230, 100, 50),  # pinky   (idx 17-20)
    ]
    # Wrist
    x, y = int(lm[0].x * w), int(lm[0].y * h)
    cv2.circle(canvas, (x, y), 6, (200, 200, 200), -1, cv2.LINE_AA)

    for finger_idx, colour in enumerate(finger_colours):
        for offset in range(4):
            lm_idx = 1 + finger_idx * 4 + offset
            x, y = int(lm[lm_idx].x * w), int(lm[lm_idx].y * h)
            r = 7 if offset == 3 else 4   # tip is larger
            cv2.circle(canvas, (x, y), r, colour, -1, cv2.LINE_AA)


def _draw_metrics_panel(canvas, face_data, hand_data):
    """Overlay live metric values in top-right corner."""
    h, w = canvas.shape[:2]
    lines = []
    if face_data:
        lines += [
            f"EAR    {face_data.ear:.4f}",
            f"Yaw    {face_data.head_yaw:+.1f} deg",
            f"Tex    {face_data.texture_score:.1f}",
            f"Blink  {'YES' if face_data.is_blinking    else 'no'}",
            f"TurnL  {'YES' if face_data.is_turned_left  else 'no'}",
            f"TurnR  {'YES' if face_data.is_turned_right else 'no'}",
        ]
    else:
        lines.append("Face  not detected")

    lines.append("")
    if hand_data:
        lines += [
            f"ThumbUp  {'YES' if hand_data.is_thumbs_up    else 'no'}",
            f"Pinky    {'YES' if hand_data.is_pinky_raised  else 'no'}",
        ]
    else:
        lines.append("Hand  not detected")

    panel_w = 210
    cv2.rectangle(canvas, (w - panel_w - 8, 4), (w - 4, 14 + len(lines) * 18),
                  (15, 15, 15), -1)

    for i, line in enumerate(lines):
        colour = (0, 255, 100) if "YES" in line else (200, 200, 200)
        cv2.putText(canvas, line, (w - panel_w, 18 + i * 18),
                    _FONT, 0.45, colour, 1, cv2.LINE_AA)


_SELECTOR_H = 36
_SELECTOR_ITEMS = [
    ("1", ChallengeType.BLINK,      "Blink"),
    ("2", ChallengeType.TURN_LEFT,  "Turn Left"),
    ("3", ChallengeType.TURN_RIGHT, "Turn Right"),
    ("4", ChallengeType.THUMBS_UP,  "Thumbs Up"),
    ("5", ChallengeType.SHOW_PINKY, "Pinky"),
    ("0", None,                     "Random"),
]


def _draw_challenge_selector(canvas, dev_challenge, w, h):
    y0 = h - _SELECTOR_H
    cv2.rectangle(canvas, (0, y0), (w, h), (18, 18, 18), -1)
    cv2.line(canvas, (0, y0), (w, y0), (70, 70, 70), 1)

    n = len(_SELECTOR_ITEMS)
    step = w // n
    for i, (key, ctype, label) in enumerate(_SELECTOR_ITEMS):
        is_selected = dev_challenge == ctype
        color = (50, 220, 50) if is_selected else (140, 140, 140)
        text = f"[{key}] {label}"
        x = i * step + 10
        if is_selected:
            (tw, th), _ = cv2.getTextSize(text, _FONT, 0.46, 1)
            cv2.rectangle(canvas, (x - 4, y0 + 6), (x + tw + 4, y0 + 6 + th + 6),
                          (0, 55, 0), -1)
        cv2.putText(canvas, text, (x, y0 + 24), _FONT, 0.46, color, 1, cv2.LINE_AA)


class DebugView:
    WINDOW = "Dev — MediaPipe Landmarks"

    def __init__(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 900, 620)
        cv2.moveWindow(self.WINDOW, 1020, 100)   # place beside main window

    def render(self, frame: np.ndarray, face_result, hand_result,
               face_data, hand_data, dev_challenge=None):
        """Draw all landmarks onto a copy of frame and display it."""
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # ── face ─────────────────────────────────────────────────────────────
        if face_result and face_result.face_landmarks:
            lm = face_result.face_landmarks[0]
            for connections, colour, thickness in _FACE_LAYERS:
                _draw_connections(canvas, lm, connections, colour, thickness, w, h)
            _draw_all_face_points(canvas, lm, w, h)

        # ── hand ─────────────────────────────────────────────────────────────
        if hand_result and hand_result.hand_landmarks:
            lm = hand_result.hand_landmarks[0]
            for connections, colour, thickness in _HAND_LAYERS:
                _draw_connections(canvas, lm, connections, colour, thickness, w, h)
            _draw_hand_points(canvas, lm, w, h)

        # ── metrics ──────────────────────────────────────────────────────────
        _draw_metrics_panel(canvas, face_data, hand_data)

        # ── header ───────────────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, 0), (w, 28), (15, 15, 15), -1)
        cv2.putText(canvas, "DEV  |  MediaPipe Landmark Viewer",
                    (10, 19), _FONT, 0.58, (100, 230, 100), 1, cv2.LINE_AA)

        # ── challenge selector bar ────────────────────────────────────────────
        _draw_challenge_selector(canvas, dev_challenge, w, h)

        cv2.imshow(self.WINDOW, canvas)

    def close(self):
        cv2.destroyWindow(self.WINDOW)
