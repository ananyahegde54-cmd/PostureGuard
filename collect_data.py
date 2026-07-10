import cv2
import mediapipe as mp
import csv
import os
import time

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pose = mp_pose.Pose()

CSV_FILE = "posture_data.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(33):
            header += [f'x{i}', f'y{i}', f'z{i}']
        header.append('label')
        writer.writerow(header)
    print("Created new CSV file!")

def collect(label, duration=120):
    name = "GOOD POSTURE" if label == 0 else "BAD POSTURE"
    print(f"\nGet ready for {name}...")
    print("Starting in 5 seconds - get into position!")
    time.sleep(5)
    print(f"Recording {name} for {duration} seconds...")

    start = time.time()
    count = 0

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        remaining = int(duration - (time.time() - start))
        cv2.putText(frame, f"{name} - {remaining}s left", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved: {count} frames", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            row = []
            for lm in results.pose_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]
            row.append(label)

            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            count += 1

        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Done! Saved {count} frames for {name}")

print("=== PostureGuard Data Collection ===")
print("We will record 2 minutes of GOOD posture, then 2 minutes of BAD posture")
print("Sit straight with good posture first!")

collect(label=0, duration=120)

print("\nNow slouch, hunch forward for BAD posture!")
collect(label=1, duration=120)

cap.release()
cv2.destroyAllWindows()
print("\nData collection complete!")
print(f"Check posture_data.csv in your PostureGuard folder")
