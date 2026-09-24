"""
PostureGuard — data collection script (v2)

Changes from the original collect_data.py:
  - Prompts for subject_id, camera_angle, and age_bracket before recording, so every
    frame is tagged with the metadata the Capture_Matrix / Subject_Registry / Session_Log
    workbook needs.
  - Records an arbitrary list of posture classes (edit POSTURE_CLASSES below), not just
    a hardcoded GOOD/BAD pair.
  - Writes two outputs per class recorded: a per-session CSV under sessions/, and an
    append to a NEW master CSV (MASTER_CSV below). It does NOT touch the old
    posture_data.csv — that file was already used for the first training run, so it's
    left alone as your frozen baseline (see Canary_Set_Manifest in the workbook).
  - After the whole run finishes, opens PostureGuard_Data_Capture_Matrix.xlsx and
    updates Session_Log, Capture_Matrix, and Subject_Registry automatically — no more
    hand-editing the spreadsheet after every session.

Run it, answer the three prompts, then follow the on-screen countdown for each class.
"""

import cv2
import mediapipe as mp
import csv
import os
import time
from datetime import datetime

import openpyxl

# ---------------------------------------------------------------------------
# CONFIG — edit these to match your setup
# ---------------------------------------------------------------------------

# (numeric label, posture_class string). The posture_class string MUST exactly match
# the posture_class values in Capture_Matrix (case-sensitive, before .upper() is applied
# for the cell_id lookup). Rename these once you've confirmed what each class actually
# means and renamed the matching rows in the workbook.
POSTURE_CLASSES = [
    (0, "class_0"),
    (1, "class_1"),
    (2, "class_2"),
]

DURATION_PER_CLASS = 60  # seconds

CAMERA_ANGLE_CODES = {
    "front": "A1",
    "left_30": "A2",
    "right_30": "A3",
    "side_90": "A4",
    "elevated_20": "A5",
}

AGE_BRACKET_CODES = {
    "teen": "T",
    "young": "Y",
    "middle": "M",
    "older": "O",
}

SESSIONS_DIR = "sessions"
MASTER_CSV = "posture_data_v2.csv"          # new file — old posture_data.csv is untouched
EXCEL_PATH = "PostureGuard_Data_Capture_Matrix.xlsx"

# Row/column layout of the workbook (matches the file already generated for this project).
CM_HEADER_ROW = 4     # Capture_Matrix header row
CM_FIRST_DATA_ROW = 5
CM_COLS = {"cell_id": 1, "camera_angle": 2, "angle_code": 3, "age_bracket": 4,
           "age_code": 5, "posture_class": 6, "target_count": 7, "current_count": 8,
           "subjects_contributed": 9, "status": 10, "notes": 11}

SR_HEADER_ROW = 4     # Subject_Registry header row
SR_FIRST_DATA_ROW = 5
SR_COLS = {"subject_id": 1, "age_bracket": 2, "age_actual": 3, "height_cm": 4,
           "gender": 5, "body_build": 6, "consent_date": 7, "first_session": 8,
           "last_session": 9, "total_frames": 10, "angles_recorded": 11, "notes": 12}

SL_HEADER_ROW = 4     # Session_Log header row
SL_FIRST_DATA_ROW = 5
SL_COLS = {"session_id": 1, "subject_id": 2, "camera_angle": 3, "angle_code": 4,
           "age_bracket": 5, "posture_class": 6, "start_time": 7, "end_time": 8,
           "duration_sec": 9, "frames_captured": 10, "frames_discarded": 11,
           "fps_actual": 12, "lighting": 13, "distance_cm": 14, "device": 15,
           "csv_path": 16, "notes": 17}

# ---------------------------------------------------------------------------
# Prompts — collected once per run of the script
# ---------------------------------------------------------------------------

def prompt_choice(label, options):
    options_list = sorted(options.keys())
    while True:
        raw = input(f"{label} ({'/'.join(options_list)}): ").strip().lower()
        if raw in options:
            return raw
        print(f"  '{raw}' is not one of {options_list}, try again.")


def prompt_text(label, default=None):
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{label}{suffix}: ").strip()
    return raw if raw else default


print("=== PostureGuard Data Collection (v2) ===")
subject_id = prompt_text("Subject ID (e.g. S001)")
camera_angle = prompt_choice("Camera angle", CAMERA_ANGLE_CODES)
age_bracket = prompt_choice("Age bracket", AGE_BRACKET_CODES)
lighting = prompt_text("Lighting (indoor_natural/indoor_artificial/mixed/low)", default="indoor_natural")
distance_cm = prompt_text("Camera-to-subject distance in cm", default="60")
device = prompt_text("Device (e.g. laptop model)", default="unknown")

angle_code = CAMERA_ANGLE_CODES[camera_angle]
age_code = AGE_BRACKET_CODES[age_bracket]

os.makedirs(SESSIONS_DIR, exist_ok=True)

master_is_new = not os.path.exists(MASTER_CSV)
if master_is_new:
    with open(MASTER_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp", "subject_id", "session_id", "camera_angle", "angle_code",
                   "age_bracket", "posture_class"]
        for i in range(33):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header += ["frame_quality", "label"]
        writer.writerow(header)
    print(f"Created new master CSV: {MASTER_CSV}")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pose = mp_pose.Pose()

# Collected across the whole run — used to update the workbook once at the end.
completed_sessions = []  # list of dicts, one per posture class recorded this run


def collect(label, posture_class, duration=DURATION_PER_CLASS):
    session_id = f"{subject_id}_{angle_code}_{age_code}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    session_csv_path = os.path.join(SESSIONS_DIR, f"{session_id}.csv")

    print(f"\nGet ready for '{posture_class}' ({camera_angle}, {age_bracket})...")
    print("Starting in 5 seconds - get into position!")
    time.sleep(5)
    print(f"Recording '{posture_class}' for {duration} seconds...")

    start_time = datetime.now()
    start = time.time()
    frames_captured = 0
    frames_discarded = 0

    with open(session_csv_path, "w", newline="") as session_f:
        session_writer = csv.writer(session_f)
        session_header = ["timestamp", "subject_id", "session_id", "camera_angle", "angle_code",
                           "age_bracket", "posture_class"]
        for i in range(33):
            session_header += [f"x{i}", f"y{i}", f"z{i}"]
        session_header += ["frame_quality", "label"]
        session_writer.writerow(session_header)

        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            remaining = int(duration - (time.time() - start))
            cv2.putText(frame, f"{posture_class} - {remaining}s left", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Saved: {frames_captured} frames", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                coords = []
                visibilities = []
                for lm in landmarks:
                    coords += [lm.x, lm.y, lm.z]
                    visibilities.append(lm.visibility)
                frame_quality = sum(visibilities) / len(visibilities)

                row = [datetime.now().isoformat(), subject_id, session_id, camera_angle,
                       angle_code, age_bracket, posture_class] + coords + [frame_quality, label]

                session_writer.writerow(row)
                with open(MASTER_CSV, "a", newline="") as master_f:
                    csv.writer(master_f).writerow(row)

                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                frames_captured += 1
            else:
                frames_discarded += 1

            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    fps_actual = round(frames_captured / elapsed, 1) if elapsed > 0 else 0.0

    print(f"Done! Saved {frames_captured} frames for '{posture_class}' "
          f"({frames_discarded} frames discarded — no landmarks detected)")

    completed_sessions.append({
        "session_id": session_id,
        "posture_class": posture_class,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": int(elapsed),
        "frames_captured": frames_captured,
        "frames_discarded": frames_discarded,
        "fps_actual": fps_actual,
        "csv_path": session_csv_path,
    })


print(f"\nWe will record {DURATION_PER_CLASS}s for each of: "
      f"{', '.join(name for _, name in POSTURE_CLASSES)}")

for label, posture_class in POSTURE_CLASSES:
    collect(label=label, posture_class=posture_class)

cap.release()
cv2.destroyAllWindows()
print("\nData collection complete!")

# ---------------------------------------------------------------------------
# Auto-update the tracking workbook
# ---------------------------------------------------------------------------

def update_workbook():
    if not os.path.exists(EXCEL_PATH):
        print(f"\nWARNING: {EXCEL_PATH} not found next to this script — "
              f"skipping workbook update. Copy it into this folder and re-run "
              f"update_workbook() manually, or log these sessions by hand:")
        for s in completed_sessions:
            print(f"  {s}")
        return

    wb = openpyxl.load_workbook(EXCEL_PATH)
    ws_matrix = wb["Capture_Matrix"]
    ws_subjects = wb["Subject_Registry"]
    ws_sessions = wb["Session_Log"]

    # --- index existing Session_Log rows so we know which (subject, cell) pairs
    # already have data, before we append this run's rows.
    existing_subject_cells = set()
    r = SL_FIRST_DATA_ROW
    while ws_sessions.cell(row=r, column=SL_COLS["session_id"]).value:
        existing_subject_id = ws_sessions.cell(row=r, column=SL_COLS["subject_id"]).value
        existing_angle_code = ws_sessions.cell(row=r, column=SL_COLS["angle_code"]).value
        existing_age_bracket = ws_sessions.cell(row=r, column=SL_COLS["age_bracket"]).value
        existing_posture = ws_sessions.cell(row=r, column=SL_COLS["posture_class"]).value
        existing_subject_cells.add((existing_subject_id, existing_angle_code,
                                     existing_age_bracket, existing_posture))
        r += 1
    next_session_row = r

    for s in completed_sessions:
        # 1) Append to Session_Log
        row_values = {
            "session_id": s["session_id"], "subject_id": subject_id,
            "camera_angle": camera_angle, "angle_code": angle_code,
            "age_bracket": age_bracket, "posture_class": s["posture_class"],
            "start_time": s["start_time"], "end_time": s["end_time"],
            "duration_sec": s["duration_sec"], "frames_captured": s["frames_captured"],
            "frames_discarded": s["frames_discarded"], "fps_actual": s["fps_actual"],
            "lighting": lighting, "distance_cm": distance_cm, "device": device,
            "csv_path": s["csv_path"], "notes": "",
        }
        for field, col in SL_COLS.items():
            ws_sessions.cell(row=next_session_row, column=col, value=row_values[field])
        next_session_row += 1

        # 2) Update Capture_Matrix — find the matching cell_id
        cell_id = f"{angle_code}_{age_code}_{s['posture_class'].upper()}"
        r = CM_FIRST_DATA_ROW
        matched = False
        while ws_matrix.cell(row=r, column=CM_COLS["cell_id"]).value:
            if ws_matrix.cell(row=r, column=CM_COLS["cell_id"]).value == cell_id:
                matched = True
                cur = ws_matrix.cell(row=r, column=CM_COLS["current_count"]).value or 0
                ws_matrix.cell(row=r, column=CM_COLS["current_count"], value=cur + s["frames_captured"])

                is_new_subject_for_cell = (subject_id, angle_code, age_bracket,
                                            s["posture_class"]) not in existing_subject_cells
                if is_new_subject_for_cell:
                    subj_count = ws_matrix.cell(row=r, column=CM_COLS["subjects_contributed"]).value or 0
                    ws_matrix.cell(row=r, column=CM_COLS["subjects_contributed"], value=subj_count + 1)
                    existing_subject_cells.add((subject_id, angle_code, age_bracket, s["posture_class"]))

                target = ws_matrix.cell(row=r, column=CM_COLS["target_count"]).value or 0
                new_current = ws_matrix.cell(row=r, column=CM_COLS["current_count"]).value
                new_subjects = ws_matrix.cell(row=r, column=CM_COLS["subjects_contributed"]).value
                if new_current >= target and new_subjects >= 3:
                    ws_matrix.cell(row=r, column=CM_COLS["status"], value="complete")
                else:
                    ws_matrix.cell(row=r, column=CM_COLS["status"], value="in_progress")
                break
            r += 1
        if not matched:
            print(f"  WARNING: no Capture_Matrix row found for cell_id '{cell_id}' — "
                  f"check that camera_angle/age_bracket/posture_class match the workbook exactly.")

    # 3) Update or create the Subject_Registry row
    r = SR_FIRST_DATA_ROW
    subject_row = None
    while ws_subjects.cell(row=r, column=SR_COLS["subject_id"]).value:
        if ws_subjects.cell(row=r, column=SR_COLS["subject_id"]).value == subject_id:
            subject_row = r
            break
        r += 1
    if subject_row is None:
        subject_row = r  # first empty row

    today = datetime.now().strftime("%Y-%m-%d")
    total_new_frames = sum(s["frames_captured"] for s in completed_sessions)

    existing_total = ws_subjects.cell(row=subject_row, column=SR_COLS["total_frames"]).value or 0
    existing_angles = ws_subjects.cell(row=subject_row, column=SR_COLS["angles_recorded"]).value or ""
    angle_list = [a for a in existing_angles.split(",") if a] if existing_angles else []
    if angle_code not in angle_list:
        angle_list.append(angle_code)

    if not ws_subjects.cell(row=subject_row, column=SR_COLS["subject_id"]).value:
        # brand-new subject
        ws_subjects.cell(row=subject_row, column=SR_COLS["subject_id"], value=subject_id)
        ws_subjects.cell(row=subject_row, column=SR_COLS["age_bracket"], value=age_bracket)
        ws_subjects.cell(row=subject_row, column=SR_COLS["consent_date"], value=today)
        ws_subjects.cell(row=subject_row, column=SR_COLS["first_session"], value=today)

    ws_subjects.cell(row=subject_row, column=SR_COLS["last_session"], value=today)
    ws_subjects.cell(row=subject_row, column=SR_COLS["total_frames"], value=existing_total + total_new_frames)
    ws_subjects.cell(row=subject_row, column=SR_COLS["angles_recorded"], value=",".join(angle_list))

    wb.save(EXCEL_PATH)
    print(f"\nWorkbook updated: {EXCEL_PATH}")
    print(f"  Session_Log: +{len(completed_sessions)} row(s)")
    print(f"  Capture_Matrix: updated counts for {len(completed_sessions)} cell(s)")
    print(f"  Subject_Registry: {subject_id} totals refreshed")


update_workbook()
print(f"\nRaw data: {MASTER_CSV} (master) and {SESSIONS_DIR}/ (per-session)")
print("Note: the original posture_data.csv was left untouched — it's your frozen "
      "baseline/canary set, not part of this run.")