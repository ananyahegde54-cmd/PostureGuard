"""
PostureGuard — one-time workbook repair script

Fixes two issues found in the committed workbook:
  1. Early sessions were logged with the old placeholder posture_class names
     (class_0/class_1/class_2) before they were renamed to good/moderate/bad.
     Capture_Matrix's cell_id lookup requires an exact string match, so those
     rows never incremented current_count — every cell still reads 0 despite
     40 real sessions in Session_Log.
  2. "s006" and "S006" were logged as two different subjects due to inconsistent
     casing when typing the Subject ID prompt.

This script rewrites Session_Log in place (normalizing posture_class and
subject_id), then rebuilds Capture_Matrix's progress columns and
Subject_Registry entirely FROM Session_Log, so the three sheets are
guaranteed consistent afterward. Safe to re-run any time — it recomputes
from scratch rather than incrementing, so it can't double-count.

collect_data.py already normalizes subject_id casing going forward (added
in this same update), so this kind of drift shouldn't recur.
"""

import openpyxl
from openpyxl.styles import Font, Border, Side
from datetime import datetime

EXCEL_PATH = "PostureGuard_Data_Capture_Matrix.xlsx"

# Old placeholder name -> current real name
LEGACY_CLASS_MAP = {
    "class_0": "good",
    "class_1": "moderate",
    "class_2": "bad",
}

CM_HEADER_ROW = 4
CM_FIRST_DATA_ROW = 5
CM_COLS = {"cell_id": 1, "camera_angle": 2, "angle_code": 3, "age_bracket": 4,
           "age_code": 5, "posture_class": 6, "target_count": 7, "current_count": 8,
           "subjects_contributed": 9, "status": 10, "notes": 11}

SR_HEADER_ROW = 4
SR_FIRST_DATA_ROW = 5
SR_COLS = {"subject_id": 1, "age_bracket": 2, "age_actual": 3, "height_cm": 4,
           "gender": 5, "body_build": 6, "consent_date": 7, "first_session": 8,
           "last_session": 9, "total_frames": 10, "angles_recorded": 11, "notes": 12}

SL_HEADER_ROW = 4
SL_FIRST_DATA_ROW = 5
SL_COLS = {"session_id": 1, "subject_id": 2, "camera_angle": 3, "angle_code": 4,
           "age_bracket": 5, "posture_class": 6, "start_time": 7, "end_time": 8,
           "duration_sec": 9, "frames_captured": 10, "frames_discarded": 11,
           "fps_actual": 12, "lighting": 13, "distance_cm": 14, "device": 15,
           "csv_path": 16, "notes": 17}

FONT = Font(name="Arial", size=10)
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def normalize_sessions(ws_sessions):
    """Fix posture_class and subject_id in place in Session_Log. Returns the
    number of rows changed for each fix, for the summary printout."""
    class_fixes = 0
    subject_fixes = 0
    r = SL_FIRST_DATA_ROW
    while ws_sessions.cell(row=r, column=SL_COLS["session_id"]).value:
        pc_cell = ws_sessions.cell(row=r, column=SL_COLS["posture_class"])
        if pc_cell.value in LEGACY_CLASS_MAP:
            pc_cell.value = LEGACY_CLASS_MAP[pc_cell.value]
            class_fixes += 1

        sid_cell = ws_sessions.cell(row=r, column=SL_COLS["subject_id"])
        if sid_cell.value:
            normalized = str(sid_cell.value).strip().upper()
            if normalized != sid_cell.value:
                sid_cell.value = normalized
                subject_fixes += 1
        r += 1
    return class_fixes, subject_fixes


def rebuild_capture_matrix(ws_matrix, ws_sessions):
    """Recompute current_count / subjects_contributed / status for every
    Capture_Matrix row from Session_Log, from scratch."""
    # cell_id -> {"frames": int, "subjects": set()}
    totals = {}
    r = SL_FIRST_DATA_ROW
    while ws_sessions.cell(row=r, column=SL_COLS["session_id"]).value:
        subject_id = ws_sessions.cell(row=r, column=SL_COLS["subject_id"]).value
        angle_code = ws_sessions.cell(row=r, column=SL_COLS["angle_code"]).value
        age_bracket = ws_sessions.cell(row=r, column=SL_COLS["age_bracket"]).value
        posture_class = ws_sessions.cell(row=r, column=SL_COLS["posture_class"]).value
        frames = ws_sessions.cell(row=r, column=SL_COLS["frames_captured"]).value or 0

        # age_code isn't stored in Session_Log directly — derive it from age_bracket
        age_code_map = {"teen": "T", "young": "Y", "middle": "M", "older": "O"}
        age_code = age_code_map.get(age_bracket, "")

        cell_id = f"{angle_code}_{age_code}_{str(posture_class).upper()}"
        entry = totals.setdefault(cell_id, {"frames": 0, "subjects": set()})
        entry["frames"] += frames
        entry["subjects"].add(subject_id)
        r += 1

    unmatched = []
    r = CM_FIRST_DATA_ROW
    while ws_matrix.cell(row=r, column=CM_COLS["cell_id"]).value:
        cell_id = ws_matrix.cell(row=r, column=CM_COLS["cell_id"]).value
        target = ws_matrix.cell(row=r, column=CM_COLS["target_count"]).value or 0
        entry = totals.pop(cell_id, None)
        if entry:
            current = entry["frames"]
            n_subjects = len(entry["subjects"])
        else:
            current = 0
            n_subjects = 0
        ws_matrix.cell(row=r, column=CM_COLS["current_count"], value=current)
        ws_matrix.cell(row=r, column=CM_COLS["subjects_contributed"], value=n_subjects)
        if current == 0:
            status = "not_started"
        elif current >= target and n_subjects >= 3:
            status = "complete"
        else:
            status = "in_progress"
        ws_matrix.cell(row=r, column=CM_COLS["status"], value=status)
        r += 1

    # any cell_ids left in `totals` had sessions logged but no matching matrix row
    unmatched = list(totals.keys())
    return unmatched


def rebuild_subject_registry(ws_subjects, ws_sessions):
    """Consolidate Subject_Registry from Session_Log, merging any duplicate-cased
    subject rows and preserving hand-entered fields (age_actual, height_cm, etc.)
    from whichever existing row had them filled in."""
    # Preserve any manually-entered extra fields, keyed by normalized subject_id
    preserved = {}
    r = SR_FIRST_DATA_ROW
    while ws_subjects.cell(row=r, column=SR_COLS["subject_id"]).value:
        raw_id = ws_subjects.cell(row=r, column=SR_COLS["subject_id"]).value
        norm_id = str(raw_id).strip().upper()
        entry = preserved.setdefault(norm_id, {})
        for field in ["age_actual", "height_cm", "gender", "body_build", "consent_date", "notes"]:
            val = ws_subjects.cell(row=r, column=SR_COLS[field]).value
            if val not in (None, ""):
                entry[field] = val
        r += 1
    last_existing_row = r  # first empty row before we clear

    # Aggregate from Session_Log
    agg = {}  # subject_id -> dict
    r = SL_FIRST_DATA_ROW
    while ws_sessions.cell(row=r, column=SL_COLS["session_id"]).value:
        subject_id = ws_sessions.cell(row=r, column=SL_COLS["subject_id"]).value
        age_bracket = ws_sessions.cell(row=r, column=SL_COLS["age_bracket"]).value
        angle_code = ws_sessions.cell(row=r, column=SL_COLS["angle_code"]).value
        frames = ws_sessions.cell(row=r, column=SL_COLS["frames_captured"]).value or 0
        start_time = ws_sessions.cell(row=r, column=SL_COLS["start_time"]).value

        entry = agg.setdefault(subject_id, {
            "age_bracket": age_bracket, "total_frames": 0, "angles": set(),
            "first_session": start_time, "last_session": start_time,
        })
        entry["total_frames"] += frames
        entry["angles"].add(angle_code)
        if start_time and (not entry["first_session"] or str(start_time) < str(entry["first_session"])):
            entry["first_session"] = start_time
        if start_time and (not entry["last_session"] or str(start_time) > str(entry["last_session"])):
            entry["last_session"] = start_time
        r += 1

    # Clear existing data rows
    for row in range(SR_FIRST_DATA_ROW, last_existing_row):
        for col in SR_COLS.values():
            ws_subjects.cell(row=row, column=col, value=None)

    # Rewrite consolidated rows, sorted by subject_id
    row = SR_FIRST_DATA_ROW
    for subject_id in sorted(agg.keys()):
        entry = agg[subject_id]
        extra = preserved.get(subject_id, {})

        def fmt_date(v):
            if v is None:
                return None
            if isinstance(v, str):
                return v.split(" ")[0]
            return v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v)

        values = {
            "subject_id": subject_id,
            "age_bracket": entry["age_bracket"],
            "age_actual": extra.get("age_actual"),
            "height_cm": extra.get("height_cm"),
            "gender": extra.get("gender"),
            "body_build": extra.get("body_build"),
            "consent_date": extra.get("consent_date"),
            "first_session": fmt_date(entry["first_session"]),
            "last_session": fmt_date(entry["last_session"]),
            "total_frames": entry["total_frames"],
            "angles_recorded": ",".join(sorted(entry["angles"])),
            "notes": extra.get("notes", ""),
        }
        for field, col in SR_COLS.items():
            cell = ws_subjects.cell(row=row, column=col, value=values[field])
            cell.font = FONT
            cell.border = BORDER
        row += 1

    return sorted(agg.keys())


def main():
    wb = openpyxl.load_workbook(EXCEL_PATH)
    ws_matrix = wb["Capture_Matrix"]
    ws_subjects = wb["Subject_Registry"]
    ws_sessions = wb["Session_Log"]

    class_fixes, subject_fixes = normalize_sessions(ws_sessions)
    print(f"Session_Log: renamed {class_fixes} legacy posture_class value(s), "
          f"normalized {subject_fixes} subject_id casing issue(s).")

    unmatched = rebuild_capture_matrix(ws_matrix, ws_sessions)
    if unmatched:
        print(f"WARNING: {len(unmatched)} cell_id(s) in Session_Log have no matching "
              f"Capture_Matrix row (check camera_angle/age_bracket/posture_class spelling): "
              f"{unmatched}")
    else:
        print("Capture_Matrix: all Session_Log rows matched a cell — counts rebuilt.")

    subjects = rebuild_subject_registry(ws_subjects, ws_sessions)
    print(f"Subject_Registry: rebuilt {len(subjects)} consolidated subject row(s): {subjects}")

    wb.save(EXCEL_PATH)
    print(f"\nSaved {EXCEL_PATH}")


if __name__ == "__main__":
    main()
