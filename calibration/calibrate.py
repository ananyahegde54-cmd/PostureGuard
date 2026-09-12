import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

# ---------------- SETTINGS ---------------- #

CALIBRATION_TIME = 30
SAVE_PATH = "calibration/baseline.json"

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

samples = []

start_time = time.time()

print("Starting calibration...")
print("Sit in your NORMAL comfortable posture.")
print("Keep your posture natural for 30 seconds.")

# ---------------- CALIBRATION ---------------- #

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    elapsed = time.time() - start_time
    remaining = max(0, CALIBRATION_TIME - elapsed)

    if results.pose_landmarks:

        landmarks = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        )

        # MediaPipe landmark indexes
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        # 1. Head Forward
        shoulder_center = (left_shoulder + right_shoulder) / 2
        head_forward = abs(nose[2] - shoulder_center[2])

        # 2. Shoulder Asymmetry
        shoulder_asym = abs(
            left_shoulder[1] - right_shoulder[1]
        )

        # 3. Spinal Offset
        hip_center = (left_hip + right_hip) / 2
        spinal_offset = np.linalg.norm(
            shoulder_center[:2] - hip_center[:2]
        )

        # 4. Neck Angle
        neck_vector = nose[:2] - shoulder_center[:2]
        vertical = np.array([0.0, -1.0])

        denominator = (
            np.linalg.norm(neck_vector) *
            np.linalg.norm(vertical)
        )

        if denominator > 0:
            cos_angle = np.dot(
                neck_vector,
                vertical
            ) / denominator

            cos_angle = np.clip(cos_angle, -1, 1)
            neck_angle = np.degrees(
                np.arccos(cos_angle)
            )
        else:
            neck_angle = 0

        # 5. Torso Lean
        torso_vector = shoulder_center[:2] - hip_center[:2]
        torso_lean = abs(torso_vector[0])

        # 6. Stability
        stability = np.mean(
            np.linalg.norm(
                landmarks[11:13, :2] -
                shoulder_center[:2],
                axis=1
            )
        )

        samples.append([
            head_forward,
            shoulder_asym,
            spinal_offset,
            neck_angle,
            torso_lean,
            stability
        ])

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # ---------------- DISPLAY ---------------- #

    cv2.putText(
        frame,
        f"CALIBRATING: {remaining:.1f}s",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        "Maintain your normal posture",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "PostureGuard Calibration",
        frame
    )

    if elapsed >= CALIBRATION_TIME:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- SAVE BASELINE ---------------- #

cap.release()
pose.close()
cv2.destroyAllWindows()

if len(samples) == 0:
    print("No calibration data collected.")
    exit()

samples = np.array(samples)

baseline = {}

feature_names = [
    "head_forward",
    "shoulder_asymmetry",
    "spinal_offset",
    "neck_angle",
    "torso_lean",
    "stability"
]

for i, name in enumerate(feature_names):
    mean = float(np.mean(samples[:, i]))
    std = float(np.std(samples[:, i]))

    baseline[name] = {
        "mean": mean,
        "std": std,
        "moderate_threshold": mean + (1.5 * std),
        "bad_threshold": mean + (3.0 * std)
    }

os.makedirs("calibration", exist_ok=True)

with open(SAVE_PATH, "w") as f:
    json.dump(baseline, f, indent=4)

print("\nCalibration completed!")
print(f"Samples collected: {len(samples)}")
print(f"Baseline saved to: {SAVE_PATH}")