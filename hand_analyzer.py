"""MediaPipe HandLandmarker wrapper for gesture-based liveness challenges.

Downloads the ~25 MB model on first run.
"""

import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
from dataclasses import dataclass

_MODEL_FILENAME = "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# 21-point hand landmark indices
_THUMB_MCP, _THUMB_TIP = 2, 4
_INDEX_PIP,  _INDEX_TIP  = 6,  8
_MIDDLE_PIP, _MIDDLE_TIP = 10, 12
_RING_PIP,   _RING_TIP   = 14, 16
_PINKY_PIP,  _PINKY_TIP  = 18, 20


def _ensure_model() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_FILENAME)
    if not os.path.exists(path):
        print(f"Downloading hand landmark model (~25 MB) to {path} ...")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("Download complete.")
    return path


def _extended(lm, tip: int, pip: int) -> bool:
    """True when finger tip is above its PIP joint (y increases downward)."""
    return lm[tip].y < lm[pip].y


def _thumbs_up(lm) -> bool:
    thumb_up     = lm[_THUMB_TIP].y < lm[_THUMB_MCP].y
    others_down  = (
        not _extended(lm, _INDEX_TIP,  _INDEX_PIP)  and
        not _extended(lm, _MIDDLE_TIP, _MIDDLE_PIP) and
        not _extended(lm, _RING_TIP,   _RING_PIP)   and
        not _extended(lm, _PINKY_TIP,  _PINKY_PIP)
    )
    return thumb_up and others_down


def _pinky_raised(lm) -> bool:
    pinky_up    = _extended(lm, _PINKY_TIP, _PINKY_PIP)
    others_down = (
        not _extended(lm, _INDEX_TIP,  _INDEX_PIP)  and
        not _extended(lm, _MIDDLE_TIP, _MIDDLE_PIP) and
        not _extended(lm, _RING_TIP,   _RING_PIP)
    )
    return pinky_up and others_down


@dataclass
class HandData:
    is_thumbs_up:    bool
    is_pinky_raised: bool


class HandAnalyzer:
    def __init__(self):
        model_path   = _ensure_model()
        base_options = _mp_python.BaseOptions(model_asset_path=model_path)
        options      = _mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = _mp_vision.HandLandmarker.create_from_options(options)
        self._last_ts    = 0
        self.last_result = None   # raw result exposed for debug view

    def analyze(self, frame_bgr: np.ndarray) -> "HandData | None":
        ts = int(time.time() * 1000)
        ts = max(ts, self._last_ts + 1)
        self._last_ts = ts

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result           = self._landmarker.detect_for_video(mp_image, ts)
        self.last_result = result

        if not result.hand_landmarks:
            return None

        lm = result.hand_landmarks[0]
        return HandData(
            is_thumbs_up=_thumbs_up(lm),
            is_pinky_raised=_pinky_raised(lm),
        )

    def close(self):
        self._landmarker.close()
