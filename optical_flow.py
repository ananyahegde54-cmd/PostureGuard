"""
PostureGuard  —  Optical Flow / Predictive Warning Module  (FR5 / FR6 / FR7)
─────────────────────────────────────────────────────────────────────────────
Drop this file next to realtime_posture.py and import it.

What it does, in plain English:
  Every frame, we compare the current greyscale image with the previous one
  using Farneback optical flow (built into OpenCV — no extra install needed).
  This gives us a "how much did pixels move" magnitude for the whole frame.

  We track that magnitude in a 30-second rolling buffer.  When the average
  starts rising AND posture is already moderate, that pattern historically
  precedes a full slouch.  We fire an early warning BEFORE bad posture
  actually starts — this is FR6.

  We also track session time and the rate of bad-posture frames to estimate
  how many minutes until the user will need a break — FR7.

Usage — add these 3 lines to realtime_posture.py:

    from optical_flow import OpticalFlowPredictor
    predictor = OpticalFlowPredictor()

    # inside the main while-loop, after you have `frame` and `current_cls`:
    warning, fatigue_msg = predictor.update(frame, current_cls)
    # warning   → None  or  a string like "Posture likely to degrade soon"
    # fatigue_msg → None  or  "Break recommended in ~4 min"
"""

import cv2
import numpy as np
from collections import deque
import time


class OpticalFlowPredictor:
    """
    Tracks micro-movement magnitude via Farneback optical flow and
    predicts posture degradation + fatigue onset.
    """

    def __init__(
        self,
        fps: float = 20.0,          # approximate webcam fps
        window_sec: float = 30.0,   # rolling window for trend analysis
        prediction_threshold: float = 0.65,  # confidence needed to warn
        break_interval_min: float = 45.0,    # target work interval before break
    ):
        self.fps         = fps
        self.window      = int(window_sec * fps)   # frames in rolling buffer
        self.pred_thresh = prediction_threshold
        self.break_min   = break_interval_min

        # ── Rolling buffers ───────────────────────────────
        self.flow_mag    = deque(maxlen=self.window)   # optical flow magnitude
        self.cls_history = deque(maxlen=self.window)   # class per frame (0/1/2)

        # ── Session state ─────────────────────────────────
        self.session_start  = time.time()
        self.bad_frame_count = 0
        self.total_frames    = 0

        # ── Previous frame for flow computation ───────────
        self.prev_gray = None

        # ── Warning cooldown (don't spam) ─────────────────
        self.last_warning_time = 0.0
        self.WARNING_COOLDOWN  = 60.0   # seconds between repeated predictions

        # ── Farneback params (tuned for webcam at ~20fps) ─
        self.fb_params = dict(
            pyr_scale  = 0.5,
            levels     = 2,
            winsize    = 12,
            iterations = 2,
            poly_n     = 5,
            poly_sigma = 1.1,
            flags      = 0,
        )

    # ─────────────────────────────────────────────────────
    def update(self, frame, current_cls: int):
        """
        Call once per frame inside the main loop.

        Args:
            frame       : raw BGR frame from cv2.VideoCapture
            current_cls : 0=good  1=moderate  2=bad  (-1=no person)

        Returns:
            warning     : str or None — predictive early-warning message
            fatigue_msg : str or None — fatigue / break countdown message
        """
        self.total_frames += 1
        if current_cls == 2:
            self.bad_frame_count += 1

        # ── Step 1: compute optical flow magnitude ────────
        magnitude = 0.0
        try:
            if frame is None or frame.size == 0:
                self.flow_mag.append(magnitude)
                self.cls_history.append(current_cls if current_cls >= 0 else 1)
                return None, None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if self.prev_gray is not None:
                if self.prev_gray.shape == gray.shape:
                    flow = cv2.calcOpticalFlowFarneback(
                        self.prev_gray, gray, None,
                        pyr_scale=0.5, levels=2, winsize=12,
                        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    magnitude = float(np.mean(mag))

            self.prev_gray = gray

        except Exception:
            self.prev_gray = None

        self.flow_mag.append(magnitude)
        self.cls_history.append(current_cls if current_cls >= 0 else 1)

        # ── Step 2: trend analysis (need ≥10s of data) ───
        warning     = None
        fatigue_msg = None

        if len(self.flow_mag) >= int(10 * self.fps):
            warning     = self._predict_degradation(current_cls)
            fatigue_msg = self._fatigue_estimate()

        return warning, fatigue_msg

    # ─────────────────────────────────────────────────────
    def _predict_degradation(self, current_cls: int):
        """
        Returns a warning string if posture is likely to degrade soon,
        else None.

        Logic:
          - Take the last 30 s of flow magnitudes.
          - Fit a linear trend (np.polyfit degree-1).
          - If slope is positive (movement increasing = fidgeting more)
            AND posture is currently good or moderate
            AND confidence score exceeds threshold
            → warn.
        """
        now = time.time()
        if now - self.last_warning_time < self.WARNING_COOLDOWN:
            return None

        # Already bad — reactive alert handles it, no need to predict
        if current_cls == 2:
            return None

        mags = np.array(self.flow_mag, dtype=np.float32)

        # Normalise 0-1 so slope is scale-independent
        m_range = mags.max() - mags.min()
        if m_range < 1e-6:
            return None   # completely still — no trend to analyse
        mags_norm = (mags - mags.min()) / m_range

        # Linear fit over whole window
        x     = np.arange(len(mags_norm))
        slope, intercept = np.polyfit(x, mags_norm, 1)

        # Recent 10-second mean vs earlier 20-second mean
        split      = int(10 * self.fps)
        recent_mag = float(np.mean(mags[-split:]))
        earlier_mag = float(np.mean(mags[:-split])) if len(mags) > split else recent_mag
        relative_increase = (recent_mag - earlier_mag) / (earlier_mag + 1e-6)

        # Confidence score: combines slope + relative recent increase
        confidence = min(1.0, max(0.0,
            0.6 * min(slope * 500, 1.0) +    # slope component
            0.4 * min(relative_increase, 1.0) # acceleration component
        ))

        if confidence >= self.pred_thresh:
            self.last_warning_time = now
            pct = int(confidence * 100)
            if current_cls == 1:
                return f"Posture likely to worsen soon  ({pct}% confidence)"
            else:
                return f"Fidgeting detected — check your posture  ({pct}% confidence)"

        return None

    # ─────────────────────────────────────────────────────
    def _fatigue_estimate(self):
        """
        Estimates minutes until a break is recommended.

        Simple heuristic:
          bad_rate = fraction of frames that were bad posture
          Estimated time-to-fatigue drops as bad_rate climbs.
          Below 10% bad → show nothing.
          Above 10% bad → count down toward break_interval_min.
        """
        if self.total_frames < 60:
            return None

        session_min = (time.time() - self.session_start) / 60.0
        bad_rate    = self.bad_frame_count / max(self.total_frames, 1)

        if bad_rate < 0.10:
            return None   # doing well, no fatigue warning

        # Adjust break interval by bad_rate: more bad posture → sooner break
        adjusted_interval = self.break_min * (1.0 - bad_rate * 0.6)
        remaining_min     = max(0.0, adjusted_interval - session_min)

        if remaining_min <= 0:
            return "Break recommended now — you've been sitting a while"
        elif remaining_min < 5:
            return f"Break recommended in ~{int(remaining_min) + 1} min"
        elif remaining_min < 15:
            return f"Break in ~{int(remaining_min)} min  (posture rate: {bad_rate*100:.0f}% bad)"
        return None

    # ─────────────────────────────────────────────────────
    def get_debug_info(self):
        """
        Returns a dict of internal state — useful for logging or
        displaying in the dashboard.
        """
        mags = list(self.flow_mag)
        session_min = (time.time() - self.session_start) / 60.0
        slope = 0.0
        if len(mags) >= 10:
            x = np.arange(len(mags))
            slope = float(np.polyfit(x, mags, 1)[0])

        return {
            "flow_mean":       round(float(np.mean(mags)) if mags else 0, 5),
            "flow_slope":      round(slope, 7),
            "bad_rate_pct":    round(100 * self.bad_frame_count /
                                     max(self.total_frames, 1), 1),
            "session_min":     round(session_min, 1),
            "buffer_fill_pct": round(100 * len(self.flow_mag) /
                                     self.window, 0),
        }
