"""MediaPipe FaceLandmarker wrapper that extracts liveness-relevant metrics per frame.

Uses the Tasks API (mediapipe >= 0.10). Downloads the ~30 MB model on first run.
"""

import os
import time
import math
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
from dataclasses import dataclass

_MODEL_FILENAME = "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# 6 landmark indices per eye for Eye Aspect Ratio (EAR).
# Order: [outer_corner, upper_outer, upper_inner, inner_corner, lower_inner, lower_outer]
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]   # anatomical left eye
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]   # anatomical right eye

_NOSE_TIP    = 1
_EYE_OUTER_L = 33    # outer corner of right eye
_EYE_OUTER_R = 263   # outer corner of left eye

# Tunable thresholds — adjust if detection feels too sensitive or sluggish
EAR_BLINK_THRESHOLD = 0.22   # EAR below this → eye is closing/closed
HEAD_TURN_THRESHOLD = 15.0   # degrees of yaw before a turn is registered


def _ensure_model() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_FILENAME)
    if not os.path.exists(path):
        print(f"Downloading face landmark model (~30 MB) to {path} ...")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("Download complete.")
    return path


@dataclass
class FaceData:
    ear: float            # average Eye Aspect Ratio (both eyes)
    head_yaw: float       # nose_x − eye_midpoint_x  (+ = right in mirrored view)
    texture_score: float  # Laplacian variance of face ROI (higher = more texture)
    face_bbox: tuple      # (x1, y1, x2, y2) pixel coords
    is_blinking: bool
    is_turned_left: bool
    is_turned_right: bool


def _compute_ear(lm: list, indices: list, w: int, h: int) -> float:
    pts = np.array([(lm[i].x * w, lm[i].y * h) for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-8)


class FaceAnalyzer:
    def __init__(self):
        model_path   = _ensure_model()
        base_options = _mp_python.BaseOptions(model_asset_path=model_path)
        options      = _mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = _mp_vision.FaceLandmarker.create_from_options(options)
        self._last_ts    = 0
        self.last_result = None   # raw result exposed for debug view

    def analyze(self, frame_bgr: np.ndarray) -> "FaceData | None":
        h, w = frame_bgr.shape[:2]

        # Ensure strictly increasing timestamp required by VIDEO mode
        ts = int(time.time() * 1000)
        ts = max(ts, self._last_ts + 1)
        self._last_ts = ts

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result           = self._landmarker.detect_for_video(mp_image, ts)
        self.last_result = result

        if not result.face_landmarks:
            return None

        lm: list = result.face_landmarks[0]   # list of NormalizedLandmark (x, y, z)

        # Eye Aspect Ratio
        left_ear  = _compute_ear(lm, _LEFT_EYE,  w, h)
        right_ear = _compute_ear(lm, _RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        # Head yaw in degrees from the 3D facial transformation matrix.
        # mat[:,2] is the face's forward vector in camera space; atan2 of its
        # X and Z components gives horizontal rotation (yaw).
        # Negated to match mirrored display: positive = turned right.
        mat = result.facial_transformation_matrixes[0]
        yaw = -math.degrees(math.atan2(mat[0][2], mat[2][2]))

        # Face bounding box
        all_x = [lm[i].x * w for i in range(len(lm))]
        all_y = [lm[i].y * h for i in range(len(lm))]
        x1 = max(0, int(min(all_x)))
        y1 = max(0, int(min(all_y)))
        x2 = min(w, int(max(all_x)))
        y2 = min(h, int(max(all_y)))

        # Passive texture score via Laplacian variance on the face ROI
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        roi  = gray[y1:y2, x1:x2]
        texture_score = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0.0

        return FaceData(
            ear=avg_ear,
            head_yaw=yaw,
            texture_score=texture_score,
            face_bbox=(x1, y1, x2, y2),
            is_blinking=avg_ear < EAR_BLINK_THRESHOLD,
            is_turned_left=yaw  < -HEAD_TURN_THRESHOLD,
            is_turned_right=yaw >  HEAD_TURN_THRESHOLD,
        )

    def close(self):
        self._landmarker.close()
