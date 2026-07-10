import cv2
import mediapipe as mp
import csv
import os
import time

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

CSV_FILE = "posture_data.csv"

# ─────────────────────────────────────────────────────────────────
#  POSTURE CLASS DEFINITIONS
#
#  label=0  GOOD       Ears stacked over shoulders. Spine upright.
#                      Shoulders level. Chin parallel to floor.
#                      Natural lumbar curve maintained.
#
#  label=1  MODERATE   The "drifting" zone — not alarming yet but
#                      trending toward bad. Show ONE OR TWO of:
#                        • Chin 2-4 cm forward of neutral
#                        • Mild upper-back rounding (soft C-curve,
#                          chest not fully collapsed)
#                        • One shoulder very slightly elevated (<3 cm)
#                        • Head tilted ~10° downward
#                        • Lower back starting to flatten (losing
#                          lumbar curve, not fully slouched yet)
#                      Think: "been at the desk 40 mins, drifting."
#
#  label=2  BAD        Clear, sustained poor posture. Show THREE+ of:
#                        • Ears noticeably in front of shoulders (>4 cm)
#                        • Pronounced thoracic hunch, collapsed chest
#                        • Head drooped toward chest OR craned forward
#                        • Obvious shoulder asymmetry (>3 cm vertical)
#                        • Pelvis posteriorly tilted — sitting on tailbone
#                      Think: "I genuinely need to stretch right now."
# ─────────────────────────────────────────────────────────────────

CLASSES = {
    0: {
        "name": "GOOD POSTURE",
        "color": (50, 220, 130),
        "cue": [
            "Ears directly above shoulders",
            "Shoulders back and level",
            "Chin parallel to the floor",
            "Natural arch in lower back",
            "Chest open — not collapsed",
        ],
    },
    1: {
        "name": "MODERATE POSTURE",
        "color": (30, 165, 255),
        "cue": [
            "Chin 2-4 cm forward of neutral",
            "Mild upper-back rounding (soft C-curve)",
            "OR one shoulder slightly higher",
            "OR head tilting ~10 degrees down",
            "Think: drifting after 40 mins of work",
        ],
    },
    2: {
        "name": "BAD POSTURE",
        "color": (60, 60, 230),
        "cue": [
            "Ears clearly in front of shoulders",
            "Full hunch — collapsed chest",
            "Head drooped OR heavily craned forward",
            "Pelvis tilted back, sitting on tailbone",
            "Think: 'I need to stretch right now'",
        ],
    },
}

# ── CSV setup ─────────────────────────────────────────────
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = [c + str(i) for i in range(33) for c in ("x", "y", "z")]
        header.append("label")
        writer.writerow(header)
    print(f"Created new {CSV_FILE}")
else:
    try:
        import pandas as pd
        existing = pd.read_csv(CSV_FILE)
        dist = existing["label"].value_counts().sort_index().to_dict()
        print(f"Appending to existing {CSV_FILE}  ({len(existing)} rows)")
        print(f"  Current distribution: {dist}")
    except Exception:
        print(f"Appending to {CSV_FILE}")


def draw_hud(frame, cls, elapsed, duration, count):
    info = CLASSES[cls]
    h, w = frame.shape[:2]
    remaining = max(0, int(duration - elapsed))
    progress  = min(1.0, elapsed / duration)

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 105), (15, 15, 20), -1)
    cv2.putText(frame, info["name"], (14, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, info["color"], 2, cv2.LINE_AA)
    cv2.putText(frame, f"Saved: {count} frames     {remaining}s remaining",
                (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 190, 190), 1, cv2.LINE_AA)

    # Progress bar
    cv2.rectangle(frame, (0, 102), (w, 107), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, 102), (int(w * progress), 107), info["color"], -1)

    # Cue panel (bottom)
    panel_h = 148
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, "POSTURE GUIDE:", (14, h - panel_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 135), 1, cv2.LINE_AA)
    for i, line in enumerate(info["cue"]):
        cv2.putText(frame, f"  {line}",
                    (14, h - panel_h + 46 + i * 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 215), 1, cv2.LINE_AA)


def countdown_preview(cls, seconds=5):
    info = CLASSES[cls]
    print(f"\n{'─'*54}")
    print(f"  Next  →  {info['name']}")
    for cue in info["cue"]:
        print(f"    • {cue}")
    print(f"{'─'*54}")
    print(f"  Get into position — starting in {seconds}s  (Q = skip phase)")

    deadline = time.time() + seconds
    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            break
        rem = int(deadline - time.time()) + 1
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 95), (15, 15, 20), -1)
        cv2.putText(frame, f"Prepare: {info['name']}",
                    (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, info["color"], 2)
        cv2.putText(frame, f"Starting in  {rem}s",
                    (14, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.imshow("PostureGuard — Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


def collect(cls, duration=120):
    countdown_preview(cls, seconds=5)
    info  = CLASSES[cls]
    start = time.time()
    count = 0

    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break

        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 240, 160), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 180, 120), thickness=2),
            )
            row = [v for lm in results.pose_landmarks.landmark
                   for v in (lm.x, lm.y, lm.z)]
            row.append(cls)
            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow(row)
            count += 1
        else:
            h, w = frame.shape[:2]
            cv2.putText(frame, "No person detected — stay in frame!",
                        (14, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 80, 255), 2, cv2.LINE_AA)

        draw_hud(frame, cls, elapsed, duration, count)
        cv2.imshow("PostureGuard — Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("  Phase skipped by user.")
            break

    print(f"  Done — saved {count} frames for {info['name']}")
    return count


# ── Main ──────────────────────────────────────────────────
print("\n╔══════════════════════════════════════════════╗")
print("║  PostureGuard  —  3-Class Data Collection    ║")
print("╠══════════════════════════════════════════════╣")
print("║  label 0  →  GOOD POSTURE      (2 minutes)  ║")
print("║  label 1  →  MODERATE POSTURE  (2 minutes)  ║")
print("║  label 2  →  BAD POSTURE       (2 minutes)  ║")
print("║                                              ║")
print("║  Press Q at any point to skip to next phase ║")
print("╚══════════════════════════════════════════════╝\n")

totals = {}
for label in [0, 1, 2]:
    totals[label] = collect(cls=label, duration=120)

cap.release()
cv2.destroyAllWindows()

print("\n╔══════════════════════════════════════╗")
print("║         Collection Complete           ║")
print("╠══════════════════════════════════════╣")
for label, count in totals.items():
    print(f"║  {CLASSES[label]['name']:<20}  {count:>5} frames  ║")
print("╚══════════════════════════════════════╝")
print(f"\n  Data saved  →  {CSV_FILE}")
