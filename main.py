"""
Liveness Detector
=================
Verifies that a webcam face belongs to a live person using:
  • Passive check  — Laplacian texture score over 20 frames (anti-photo / anti-screen)
  • Active challenges — 2 random challenges: blink / turn left / turn right / thumbs up / pinky

Usage
-----
  python main.py               # auto-selects camera
  python main.py --camera 1    # use camera index 1
  python main.py --list        # list available cameras
  python main.py --dev         # open landmark debug window alongside main window

Controls
--------
  Q  quit
  R  restart (new set of random challenges)
"""

import sys
import platform
import collections

try:
    import cv2
    import mediapipe  # noqa: F401
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install with:  pip install -r requirements.txt")
    sys.exit(1)


def _camera_backend() -> int:
    """Return the best OpenCV camera backend for the current OS."""
    system = platform.system()
    if system == "Darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    elif system == "Windows":
        return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    else:
        return cv2.CAP_ANY

from face_analyzer import FaceAnalyzer
from hand_analyzer import HandAnalyzer
from challenges import ChallengeRunner, ChallengeType
from debug_view import DebugView
import display

_DEV_KEYS: dict[int, "ChallengeType | None"] = {
    ord("1"): ChallengeType.BLINK,
    ord("2"): ChallengeType.TURN_LEFT,
    ord("3"): ChallengeType.TURN_RIGHT,
    ord("4"): ChallengeType.THUMBS_UP,
    ord("5"): ChallengeType.SHOW_PINKY,
    ord("0"): None,  # reset to random
}

_DEV_LABELS: dict["ChallengeType | None", str] = {
    ChallengeType.BLINK:      "Blink",
    ChallengeType.TURN_LEFT:  "Turn Left",
    ChallengeType.TURN_RIGHT: "Turn Right",
    ChallengeType.THUMBS_UP:  "Thumbs Up",
    ChallengeType.SHOW_PINKY: "Pinky",
    None:                     "Random",
}

NUM_CHALLENGES        = 2
PASSIVE_FRAMES_NEEDED = 20
PASSIVE_TEXTURE_WARN  = 50.0


def list_cameras():
    print("Scanning for cameras...")
    backend = _camera_backend()
    found = []
    for i in range(6):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            found.append((i, w, h))
    if not found:
        print("  No cameras found.")
    else:
        print(found)
        for idx, w, h in found:
            tag = " ← built-in (likely)" if h <= 720 else ""
            print(f"  [{idx}]  {w}x{h}{tag}")
    return found


def pick_camera(cameras):
    """Return the index most likely to be the built-in/primary camera."""
    if not cameras:
        return 0
    # Prefer 720p built-in over higher-resolution external cameras
    for idx, w, h in cameras:
        if h == 720:
            return idx
    return cameras[-1][0]


def main():
    args = sys.argv[1:]

    dev_mode = "--dev" in args

    if "--list" in args:
        list_cameras()
        return

    # Determine camera index
    if "--camera" in args:
        try:
            cam_idx = int(args[args.index("--camera") + 1])
        except (IndexError, ValueError):
            print("Usage: python main.py --camera <index>")
            sys.exit(1)
    else:
        cameras = list_cameras()
        print(f"Found {len(cameras)} cameras.")
        print(cameras)
        cam_idx = pick_camera(cameras)
        print(f"Using camera [{cam_idx}]. Pass --camera <index> to override.")

    cap = cv2.VideoCapture(cam_idx, _camera_backend())
    if not cap.isOpened():
        print(f"Error: cannot open camera [{cam_idx}].")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    analyzer      = FaceAnalyzer()
    hand_analyzer = HandAnalyzer()
    runner        = ChallengeRunner(NUM_CHALLENGES)
    texture_buf = collections.deque(maxlen=PASSIVE_FRAMES_NEEDED)

    state            = "NO_FACE"
    verdict          = None
    low_texture_warn = False
    dev_challenge: "ChallengeType | None" = None  # None = use random challenges

    cv2.namedWindow("Liveness Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Liveness Detector", 900, 620)
    cv2.moveWindow("Liveness Detector", 100, 100)
    cv2.setWindowProperty("Liveness Detector", cv2.WND_PROP_TOPMOST, 1)

    debug = DebugView() if dev_mode else None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # transient camera hiccup

        frame     = cv2.flip(frame, 1)
        face_data = analyzer.analyze(frame)
        hand_data = hand_analyzer.analyze(frame)

        if verdict is None:
            if face_data is None:
                if state != "NO_FACE":
                    texture_buf.clear()
                state = "NO_FACE"
            else:
                texture_buf.append(face_data.texture_score)

                if state == "NO_FACE":
                    texture_buf.clear()
                    texture_buf.append(face_data.texture_score)
                    low_texture_warn = False
                    if dev_mode and dev_challenge is not None:
                        runner = ChallengeRunner(types=[dev_challenge])
                        state  = "CHALLENGE"
                    else:
                        runner = ChallengeRunner(NUM_CHALLENGES)
                        state  = "PASSIVE"

                elif state == "PASSIVE":
                    if len(texture_buf) >= PASSIVE_FRAMES_NEEDED:
                        avg = sum(texture_buf) / len(texture_buf)
                        if avg < PASSIVE_TEXTURE_WARN:
                            low_texture_warn = True
                        state = "CHALLENGE"

                elif state == "CHALLENGE":
                    if runner.current and runner.current.is_expired:
                        verdict = "FAILED"
                        state   = "FAILED"
                    elif runner.update(face_data, hand_data) and runner.all_done:
                        verdict = "LIVE"
                        state   = "PASSED"

        avg_texture = sum(texture_buf) / len(texture_buf) if texture_buf else 0.0
        dev_label = _DEV_LABELS[dev_challenge] if dev_mode else None
        display.render(frame, face_data, state, runner,
                       avg_texture, verdict, low_texture_warn, hand_data,
                       dev_label=dev_label)
        cv2.imshow("Liveness Detector", frame)

        if debug:
            debug.render(frame, analyzer.last_result, hand_analyzer.last_result,
                         face_data, hand_data, dev_challenge)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            if dev_mode and dev_challenge is not None:
                runner = ChallengeRunner(types=[dev_challenge])
            else:
                runner = ChallengeRunner(NUM_CHALLENGES)
            texture_buf.clear()
            state            = "NO_FACE"
            verdict          = None
            low_texture_warn = False
        if dev_mode and key in _DEV_KEYS:
            dev_challenge = _DEV_KEYS[key]
            if dev_challenge is not None:
                runner = ChallengeRunner(types=[dev_challenge])
                state  = "CHALLENGE"
            else:
                runner = ChallengeRunner(NUM_CHALLENGES)
                state  = "NO_FACE"
            texture_buf.clear()
            verdict          = None
            low_texture_warn = False

    cap.release()
    analyzer.close()
    hand_analyzer.close()
    if debug:
        debug.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
