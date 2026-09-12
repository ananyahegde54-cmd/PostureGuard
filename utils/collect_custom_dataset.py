import cv2
import mediapipe as mp
import numpy as np
import os

CLASS_NAME = "slouch"   # change: good / bad / slouch
SAMPLE_NO = 20         # change from 1 to 20

save_folder = f"data/new_data/{CLASS_NAME}"
os.makedirs(save_folder, exist_ok=True)

save_path = os.path.join(save_folder, f"{CLASS_NAME}_{SAMPLE_NO}.npy")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = []
        for lm in result.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        sequence.append(landmarks)

        cv2.putText(
            frame,
            f"{CLASS_NAME.upper()} {len(sequence)}/30",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Collect Custom Dataset", frame)

    if len(sequence) == 30:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

sequence = np.array(sequence)
np.save(save_path, sequence)

print("Saved:", save_path, sequence.shape)