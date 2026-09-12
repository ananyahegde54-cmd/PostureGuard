import cv2
import mediapipe as mp
import numpy as np

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
        indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]

        for idx in indices:
             lm = result.pose_landmarks.landmark[idx]
             landmarks.append([lm.x, lm.y, lm.z])

        sequence.append(landmarks)

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Collecting Landmarks", frame)

    if len(sequence) == 30:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

sequence = np.array(sequence)
print("Collected shape:", sequence.shape)

np.save("data/leaning.npy", sequence)
print("Saved to data/sample_sequence.npy")