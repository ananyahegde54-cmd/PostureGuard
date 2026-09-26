"""
PostureGuard — data collection script (simple version)

Records three posture classes into a single fresh CSV:
  0 = good
  1 = moderate
  2 = bad

No subject/angle/age prompts, no workbook syncing — just run it, follow the
on-screen countdown for each class, and it writes straight to CSV_FILE in the
same column format your existing train_lstm.py / realtime_posture.py already
expect (x0,y0,z0,...,x32,y32,z32,label). Re-run it as many times as you want
to add more data for any class — it always appends, never overwrites.
"""

import cv2
import mediapipe as mp
import csv
import os
import time

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pose = mp_pose.Pose()

CSV_FILE = "posture_data_final.csv"   # fresh file — doesn't touch posture_data.csv or posture_data_v2.csv

DURATION_PER_CLASS = 60  # seconds — change this if you want longer/shorter recordings

CLASSES = [
    (0, "GOOD posture"),
    (1, "MODERATE posture"),
    (2, "BAD posture"),
]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(33):
            header += [f'x{i}', f'y{i}', f'z{i}']
        header.append('label')
        writer.writerow(header)
    print(f"Created new CSV file: {CSV_FILE}")
else:
    print(f"Appending to existing CSV file: {CSV_FILE}")


def collect(label, name, duration=DURATION_PER_CLASS):
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


print("=== PostureGuard Data Collection (simple) ===")
print(f"We will record {DURATION_PER_CLASS}s each for: "
      f"{', '.join(name for _, name in CLASSES)}")
print("Sit straight with good posture first, then progressively worse for each class.")

for label, name in CLASSES:
    collect(label=label, name=name)

cap.release()
cv2.destroyAllWindows()
print("\nData collection complete!")
print(f"Check {CSV_FILE} in your PostureGuard folder")
