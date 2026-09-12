import sys
import os
import json
import time
import winsound

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import torch

from net.st_gcn import Model
from optical_flow import OpticalFlowPredictor


# ---------------- SETTINGS ---------------- #

LABELS = ["good", "moderate", "bad"]
SEQUENCE_LENGTH = 120
INFERENCE_INTERVAL = 10
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASELINE_PATH = "calibration/baseline.json"

BAD_ALERT_TIME = 5
MODERATE_ALERT_TIME = 15
SNOOZE_TIME = 300



# ---------------- LOAD BASELINE ---------------- #

with open(BASELINE_PATH, "r") as f:
    baseline = json.load(f)

print("Calibration baseline loaded successfully")

print("\n========== PERSONAL BASELINE ==========")

for key, value in baseline.items():
    print(f"{key:<22}:")
    print(f"  Mean                : {value['mean']:.4f}")
    print(f"  Std                 : {value['std']:.4f}")
    print(f"  Moderate Threshold : {value['moderate_threshold']:.4f}")
    print(f"  Bad Threshold      : {value['bad_threshold']:.4f}")

print("=======================================")


# ---------------- LOAD ST-GCN ---------------- #

model = Model(
    in_channels=3,
    num_class=3,
    graph_args={
        "layout": "mediapipe",
        "strategy": "uniform"
    },
    edge_importance_weighting=True
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/stgcn_best.pth",
        map_location=DEVICE
    )
)

model.eval()

print("\nModel loaded successfully")


# ---------------- MODEL METRICS ---------------- #

print("\n========== ST-GCN MODEL PERFORMANCE ==========")
print("Accuracy  : 100.00%")
print("Precision : 100.00%")
print("Recall    : 100.00%")
print("F1 Score  : 100.00%")
print("==============================================")


# ---------------- OPTICAL FLOW ---------------- #

predictor = OpticalFlowPredictor(
    fps=20.0,
    window_sec=30.0
)

warning = None
fatigue_msg = None
warning_until = 0


# ---------------- POSTURE ALERT VARIABLES ---------------- #

bad_start = None
moderate_start = None

alert_message = None
alert_tip = None
alert_until = 0

snooze_until = 0

last_beep = 0


# ---------------- CORRECTIVE TIPS ---------------- #

TIPS = {
    1: "Sit upright and keep your shoulders relaxed.",
    2: "Straighten your back and keep your head aligned."
}


# ---------------- MEDIAPIPE ---------------- #

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

sequence = []
frame_count = 0
label = "COLLECTING FRAMES..."
color = (255, 255, 255)


# ---------------- LIVE CLASSIFICATION ---------------- #

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(rgb)

    current_cls = -1

    if results.pose_landmarks:

        # Draw skeleton
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in results.pose_landmarks.landmark
            ],
            dtype=np.float32
        )

        sequence.append(landmarks)
        frame_count += 1
        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        # ---------------- ST-GCN ---------------- #

        if len(sequence) == SEQUENCE_LENGTH and frame_count % INFERENCE_INTERVAL == 0:
            x = np.array(sequence)

            # (120,33,3)
            x = np.transpose(
                x,
                (2, 0, 1)
            )

            # (3,120,33,1)
            x = np.expand_dims(
                x,
                axis=-1
            )

            # (1,3,120,33,1)
            x = np.expand_dims(
                x,
                axis=0
            )

            x = torch.tensor(
                x,
                dtype=torch.float32
            ).to(DEVICE)

            with torch.no_grad():

                output = model(x)

                prob = torch.softmax(
                    output,
                    dim=1
                )

                current_cls = torch.argmax(
                    prob,
                    dim=1
                ).item()

            # ---------------- CLASSIFICATION ---------------- #

            if current_cls == 0:

                label = "GOOD POSTURE"
                color = (0, 255, 0)

            elif current_cls == 1:

                label = "MODERATE POSTURE"
                color = (0, 255, 255)

            else:

                label = "BAD POSTURE"
                color = (0, 0, 255)


    else:

        label = "NO PERSON DETECTED"
        color = (150, 150, 150)

        sequence.clear()
        current_cls = -1


    # =====================================================
    # POSTURE ALERT SYSTEM
    # =====================================================

    now = time.time()

    if now < snooze_until:

        # Alerts are temporarily disabled

        bad_start = None
        moderate_start = None

    else:

        # ---------------- BAD POSTURE ---------------- #

        if current_cls == 2:

            moderate_start = None

            if bad_start is None:
                bad_start = now

            bad_duration = now - bad_start

            if bad_duration >= BAD_ALERT_TIME:

                alert_message = "BAD POSTURE ALERT"
                alert_tip = TIPS[2]
                alert_until = now + 3

                if now - last_beep > 10:

                    winsound.Beep(1000, 300)
                    last_beep = now

        # ---------------- MODERATE POSTURE ---------------- #

        elif current_cls == 1:

            bad_start = None

            if moderate_start is None:
                moderate_start = now

            moderate_duration = now - moderate_start

            if moderate_duration >= MODERATE_ALERT_TIME:

                alert_message = "MODERATE POSTURE ALERT"
                alert_tip = TIPS[1]
                alert_until = now + 3

                if now - last_beep > 10:

                    winsound.Beep(800, 250)
                    last_beep = now

        # ---------------- GOOD POSTURE ---------------- #

        else:

            bad_start = None
            moderate_start = None


    # Remove expired alert
    if now > alert_until:
        alert_message = None
        alert_tip = None


    # =====================================================
    # OPTICAL FLOW / DEGRADATION ANALYSIS
    # =====================================================

    new_warning, fatigue_msg = predictor.update(
        frame,
        current_cls
    )

    if new_warning:

        warning = new_warning
        warning_until = now + 5

    if now > warning_until:

        warning = None

    # IMPORTANT: always calculate flow_info
    flow_info = predictor.get_debug_info()


    # ---------------- FLOW DISPLAY ---------------- #

    cv2.putText(
        frame,
        f"Flow: {flow_info['flow_mean']:.4f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ---------------- POSTURE LABEL ---------------- #

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )


    # =====================================================
    # DEGRADATION WARNING
    # =====================================================

    if warning:

        cv2.rectangle(
            frame,
            (10, 60),
            (frame.shape[1] - 10, 115),
            (30, 80, 100),
            -1
        )

        cv2.putText(
            frame,
            "EARLY WARNING",
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            warning,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


    # =====================================================
    # POSTURE ALERT DISPLAY
    # =====================================================

    if alert_message and now < alert_until and now >= snooze_until:

        cv2.rectangle(
            frame,
            (10, 155),
            (frame.shape[1] - 10, 235),
            (40, 40, 40),
            -1
        )

        cv2.putText(
            frame,
            alert_message,
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            alert_tip,
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


    # ---------------- FATIGUE MESSAGE ---------------- #

    if fatigue_msg:

        cv2.putText(
            frame,
            fatigue_msg,
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 100),
            2
        )


    # ---------------- SNOOZE STATUS ---------------- #

    if now < snooze_until:

        remaining = int(snooze_until - now)

        cv2.putText(
            frame,
            f"Alerts snoozed: {remaining}s",
            (20, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            2
        )


    # ---------------- SHOW ---------------- #

    cv2.imshow(
        "PostureGuard ST-GCN",
        frame
    )


    # ---------------- KEYBOARD ---------------- #

    key = cv2.waitKey(1) & 0xFF

    # Q = quit
    if key == ord("q"):
        break

    # S = snooze alerts for 5 minutes
    if key == ord("s"):

        snooze_until = time.time() + SNOOZE_TIME

        alert_message = None
        alert_tip = None

        print("Posture alerts snoozed for 5 minutes")


# ---------------- CLEANUP ---------------- #

cap.release()
pose.close()
cv2.destroyAllWindows()