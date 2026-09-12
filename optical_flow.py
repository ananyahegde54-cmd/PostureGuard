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
        fps: float = 20.0,
        window_sec: float = 30.0,
        prediction_threshold: float = 0.30,
        break_interval_min: float = 45.0,
    ):
        self.fps = fps
        self.window = int(window_sec * fps)
        self.pred_thresh = prediction_threshold
        self.break_min = break_interval_min

        self.flow_mag = deque(maxlen=self.window)
        self.cls_history = deque(maxlen=self.window)

        self.session_start = time.time()
        self.bad_frame_count = 0
        self.total_frames = 0

        self.prev_gray = None

        self.last_warning_time = 0.0
        self.WARNING_COOLDOWN = 10.0

        self.fb_params = dict(
            pyr_scale=0.5,
            levels=2,
            winsize=12,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )

    def update(self, frame, current_cls: int):

        self.total_frames += 1

        if current_cls == 2:
            self.bad_frame_count += 1

        magnitude = 0.0

        try:
            if frame is None or frame.size == 0:
                self.flow_mag.append(magnitude)
                self.cls_history.append(
                    current_cls if current_cls >= 0 else 1
                )
                return None, None

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

            if self.prev_gray is not None:

                if self.prev_gray.shape == gray.shape:

                    flow = cv2.calcOpticalFlowFarneback(
                        self.prev_gray,
                        gray,
                        None,
                        pyr_scale=0.5,
                        levels=2,
                        winsize=12,
                        iterations=2,
                        poly_n=5,
                        poly_sigma=1.1,
                        flags=0
                    )

                    mag, _ = cv2.cartToPolar(
                        flow[..., 0],
                        flow[..., 1]
                    )

                    magnitude = float(np.mean(mag))

                    # Ignore tiny camera/background movements
                    if magnitude < 0.05:
                        magnitude = 0.0

            self.prev_gray = gray

        except Exception:

            self.prev_gray = None

        self.flow_mag.append(magnitude)

        self.cls_history.append(
            current_cls if current_cls >= 0 else 1
        )

        warning = None
        fatigue_msg = None

        if len(self.flow_mag) >= int(10 * self.fps):

            warning = self._predict_degradation(
                current_cls
            )

            fatigue_msg = self._fatigue_estimate()

        return warning, fatigue_msg

    def _predict_degradation(self, current_cls: int):

        now = time.time()

        if now - self.last_warning_time < self.WARNING_COOLDOWN:
            return None

        if current_cls == 2:
            return None

        mags = np.array(
            self.flow_mag,
            dtype=np.float32
        )

        m_range = mags.max() - mags.min()

        if m_range < 1e-6:
            return None

        mags_norm = (
            mags - mags.min()
        ) / m_range

        x = np.arange(
            len(mags_norm)
        )

        slope, intercept = np.polyfit(
            x,
            mags_norm,
            1
        )

        split = int(
            10 * self.fps
        )

        recent_mag = float(
            np.mean(mags[-split:])
        )

        earlier_mag = (
            float(np.mean(mags[:-split]))
            if len(mags) > split
            else recent_mag
        )

        relative_increase = (
            (recent_mag - earlier_mag)
            / (earlier_mag + 1e-6)
        )

        confidence = min(
            1.0,
            max(
                0.0,
                0.6 * min(
                    slope * 500,
                    1.0
                )
                +
                0.4 * min(
                    relative_increase,
                    1.0
                )
            )
        )

        if confidence >= self.pred_thresh:

            self.last_warning_time = now

            pct = int(
                confidence * 100
            )

            if current_cls == 1:

                return (
                    f"Posture likely to worsen soon "
                    f"({pct}% confidence)"
                )

            else:

                return (
                    f"Fidgeting detected - check your posture "
                    f"({pct}% confidence)"
                )

        return None

    def _fatigue_estimate(self):

        if self.total_frames < 60:
            return None

        session_min = (
            time.time() -
            self.session_start
        ) / 60.0

        bad_rate = (
            self.bad_frame_count /
            max(self.total_frames, 1)
        )

        if bad_rate < 0.10:
            return None

        adjusted_interval = (
            self.break_min *
            (1.0 - bad_rate * 0.6)
        )

        remaining_min = max(
            0.0,
            adjusted_interval -
            session_min
        )

        if remaining_min <= 0:

            return (
                "Break recommended now — "
                "you've been sitting a while"
            )

        elif remaining_min < 5:

            return (
                f"Break recommended in "
                f"~{int(remaining_min) + 1} min"
            )

        elif remaining_min < 15:

            return (
                f"Break in ~{int(remaining_min)} min "
                f"(posture rate: "
                f"{bad_rate * 100:.0f}% bad)"
            )

        return None

    def get_debug_info(self):

        mags = list(
            self.flow_mag
        )

        session_min = (
            time.time() -
            self.session_start
        ) / 60.0

        slope = 0.0

        if len(mags) >= 10:

            x = np.arange(
                len(mags)
            )

            slope = float(
                np.polyfit(
                    x,
                    mags,
                    1
                )[0]
            )

        return {
            "flow_mean": round(
                float(np.mean(mags))
                if mags else 0,
                5
            ),

            "flow_slope": round(
                slope,
                7
            ),

            "bad_rate_pct": round(
                100 *
                self.bad_frame_count /
                max(self.total_frames, 1),
                1
            ),

            "session_min": round(
                session_min,
                1
            ),

            "buffer_fill_pct": round(
                100 *
                len(self.flow_mag) /
                self.window,
                0
            ),
        }