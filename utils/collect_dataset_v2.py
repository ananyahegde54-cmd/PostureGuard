import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---------------- SETTINGS ---------------- #

DATASET_PATH = "data/new_data"

CLASSES = {
    ord('g'): "good",
    ord('m'): "moderate",
    ord('b'): "bad"
}

FRAMES_PER_SAMPLE = 120

# ------------------------------------------ #

# Create folders if they don't exist
for folder in CLASSES.values():
    os.makedirs(os.path.join(DATASET_PATH, folder), exist_ok=True)

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# ---------------- HELPER FUNCTIONS ---------------- #

def get_next_filename(folder, prefix):
    """
    Returns:
        good_001.npy
        good_002.npy
        ...
    """
    files = [
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ]

    if len(files) == 0:
        return f"{prefix}_001.npy"

    nums = []

    for f in files:
        try:
            num = int(f.split("_")[-1].split(".")[0])
            nums.append(num)
        except:
            pass

    next_num = max(nums) + 1 if nums else 1

    return f"{prefix}_{next_num:03d}.npy"


def countdown(frame):

    for sec in [3, 2, 1]:

        start = time.time()

        while time.time() - start < 1:

            temp = frame.copy()

            cv2.putText(
                temp,
                f"Starting in {sec}",
                (150, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

            cv2.imshow("Dataset Collector", temp)

            cv2.waitKey(1)


def record_sample(class_name):

    print(f"\nRecording {class_name}...")

    sequence = []

    while len(sequence) < FRAMES_PER_SAMPLE:

        ret, frame = cap.read()

        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        if results.pose_landmarks:

            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            landmarks = []

            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            sequence.append(landmarks)

        cv2.putText(
            frame,
            f"{class_name.upper()}  {len(sequence)}/{FRAMES_PER_SAMPLE}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

        cv2.imshow("Dataset Collector", frame)

        cv2.waitKey(1)

    sequence = np.array(sequence, dtype=np.float32)

    folder = os.path.join(DATASET_PATH, class_name)

    filename = get_next_filename(folder, class_name)

    np.save(
        os.path.join(folder, filename),
        sequence
    )

    print(f"Saved -> {filename}")
    # ---------------- MAIN LOOP ---------------- #

print("\n===================================")
print(" Dataset Collector Started")
print("===================================")
print("Press G -> Good")
print("Press M -> Moderate")
print("Press B -> Bad")
print("Press Q -> Quit")
print("===================================\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    display = frame.copy()

    cv2.putText(
        display,
        "G=Good  M=Moderate  B=Bad  Q=Quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Dataset Collector", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key in CLASSES:

        class_name = CLASSES[key]

        # Freeze current frame for countdown
        countdown(display)

        record_sample(class_name)

cap.release()
cv2.destroyAllWindows()