"""
hand_tracker.py
---------------
MediaPipe-based hand tracker with:
  - EMA-smoothed index-finger-tip cursor
  - Scale-invariant pinch detection with hysteresis thresholding
  - Fist detection (3+ fingers curled) to trigger game start
"""

import mediapipe as mp
import numpy as np


class HandTracker:
    """Detects a single hand and exposes a smoothed cursor + pinch + fist state."""

    # EMA smoothing factor
    ALPHA: float = 0.2

    # Pinch thresholds (normalized distance)
    GRAB_THRESHOLD: float    = 0.15
    RELEASE_THRESHOLD: float = 0.22

    # Fist: how many fingers must be curled (out of 4)
    FIST_FINGER_COUNT: int = 3

    # MediaPipe landmark indices
    _INDEX_TIP  = 8
    _THUMB_TIP  = 4
    _WRIST      = 0
    _MIDDLE_MCP = 9

    # Fingertip / MCP pairs for fist detection
    _FINGER_TIPS = [8, 12, 16, 20]
    _FINGER_MCPS = [5,  9, 13, 17]

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        self._smooth_x: float | None = None
        self._smooth_y: float | None = None

        self.is_pinching: bool = False
        self.is_fist:     bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_rgb: np.ndarray):
        """
        Process a single RGB frame.

        Returns
        -------
        cursor      : (x, y) pixel coords, or None if no hand detected
        is_pinching : grab state
        is_fist     : fist state (all fingers curled)
        """
        h, w = frame_rgb.shape[:2]
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            self.is_fist = False
            return None, self.is_pinching, self.is_fist

        landmarks = results.multi_hand_landmarks[0].landmark

        # --- EMA-smoothed cursor (index fingertip) ---
        raw_x = landmarks[self._INDEX_TIP].x * w
        raw_y = landmarks[self._INDEX_TIP].y * h

        if self._smooth_x is None:
            self._smooth_x, self._smooth_y = raw_x, raw_y
        else:
            self._smooth_x = self.ALPHA * raw_x + (1 - self.ALPHA) * self._smooth_x
            self._smooth_y = self.ALPHA * raw_y + (1 - self.ALPHA) * self._smooth_y

        cursor = (int(self._smooth_x), int(self._smooth_y))

        # --- Pinch detection ---
        norm_dist = self._normalized_pinch_distance(landmarks, w, h)
        self._update_pinch_state(norm_dist)

        # --- Fist detection ---
        self.is_fist = self._detect_fist(landmarks)

        return cursor, self.is_pinching, self.is_fist

    def close(self):
        self._hands.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalized_pinch_distance(self, landmarks, w: int, h: int) -> float:
        def px(lm):
            return np.array([lm.x * w, lm.y * h])

        index_tip  = px(landmarks[self._INDEX_TIP])
        thumb_tip  = px(landmarks[self._THUMB_TIP])
        wrist      = px(landmarks[self._WRIST])
        middle_mcp = px(landmarks[self._MIDDLE_MCP])

        pinch_dist = float(np.linalg.norm(index_tip - thumb_tip))
        scale_dist = float(np.linalg.norm(wrist - middle_mcp))

        if scale_dist < 1e-6:
            return 1.0
        return pinch_dist / scale_dist

    def _update_pinch_state(self, norm_dist: float):
        if not self.is_pinching and norm_dist < self.GRAB_THRESHOLD:
            self.is_pinching = True
        elif self.is_pinching and norm_dist > self.RELEASE_THRESHOLD:
            self.is_pinching = False

    def _detect_fist(self, landmarks) -> bool:
        """Return True when at least FIST_FINGER_COUNT fingers are curled down."""
        curled = sum(
            1 for tip, mcp in zip(self._FINGER_TIPS, self._FINGER_MCPS)
            if landmarks[tip].y > landmarks[mcp].y
        )
        return curled >= self.FIST_FINGER_COUNT
