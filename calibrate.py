"""
PostureGuard  —  Personal Baseline Calibration  (FR8 / FR9)
────────────────────────────────────────────────────────────
Records 60 seconds of the user's comfortable upright posture,
computes their personal averages for all 5 biomechanical metrics,
and saves them to  user_baseline.json.

realtime_posture.py loads this file at startup and offsets its
warning thresholds so alerts are personalised, not generic.

Run once per user (or whenever they want to recalibrate).
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

BASELINE_FILE  = "user_baseline.json"
CALIB_DURATION = 60      # seconds of recording
WARMUP         = 5       # countdown before recording starts
MIN_FRAMES     = 100     # minimum valid frames needed

# ── Metric computation (mirrors realtime_posture.py) ─────
def compute_metrics(lms):
    def pt(i):
        return np.array([lms[i].x, lms[i].y, lms[i].z])

    ear_mid    = (pt(7)  + pt(8))  / 2
    sh_mid     = (pt(11) + pt(12)) / 2
    hip_mid    = (pt(23) + pt(24)) / 2
    l_sh, r_sh = pt(11), pt(12)

    head_forward   = float(sh_mid[0] - ear_mid[0])
    shoulder_asym  = float(abs(l_sh[1] - r_sh[1]))
    spinal_offset  = float(abs(sh_mid[0] - hip_mid[0]))

    vec_neck  = sh_mid[:2] - ear_mid[:2]
    neck_angle = float(np.degrees(
        np.arctan2(abs(vec_neck[0]), abs(vec_neck[1]) + 1e-6)
    ))

    vec_torso  = hip_mid[:2] - sh_mid[:2]
    torso_lean = float(np.degrees(
        np.arctan2(abs(vec_torso[0]), abs(vec_torso[1]) + 1e-6)
    ))

    return {
        "head_forward":   head_forward,
        "shoulder_asym":  shoulder_asym,
        "spinal_offset":  spinal_offset,
        "neck_angle_deg": neck_angle,
        "torso_lean_deg": torso_lean,
    }


# ── Drawing helpers ───────────────────────────────────────
def draw_bar(frame, x, y, w, h, value, vmin, vmax, color, label):
    """Draws a small horizontal progress bar with label."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 35, 40), -1)
    pct   = max(0.0, min(1.0, (value - vmin) / (vmax - vmin + 1e-6)))
    cv2.rectangle(frame, (x, y), (x + int(w * pct), y + h), color, -1)
    cv2.putText(frame, f"{label}: {value:+.3f}",
                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 200, 200), 1, cv2.LINE_AA)


def draw_hud(frame, phase, elapsed, duration, count, metrics, stability):
    h_f, w_f = frame.shape[:2]

    # ── Top bar ──────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w_f, 110), (12, 14, 18), -1)

    if phase == "countdown":
        rem = int(duration - elapsed) + 1
        cv2.putText(frame, "Get into your NATURAL COMFORTABLE posture",
                    (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                    (0, 220, 130), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Recording starts in  {rem}s",
                    (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (180, 180, 180), 1, cv2.LINE_AA)

    elif phase == "recording":
        remaining = int(duration - elapsed)
        progress  = min(1.0, elapsed / duration)
        cv2.putText(frame, "CALIBRATING  —  hold your natural posture",
                    (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                    (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,
                    f"Frames: {count}     Remaining: {remaining}s     "
                    f"Stability: {stability:.0f}%",
                    (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (180, 180, 180), 1, cv2.LINE_AA)
        # Progress bar
        cv2.rectangle(frame, (0, 106), (w_f, 110), (35, 35, 40), -1)
        cv2.rectangle(frame, (0, 106), (int(w_f * progress), 110),
                      (0, 200, 255), -1)

    elif phase == "done":
        cv2.putText(frame, "Calibration complete!",
                    (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 220, 130), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Captured {count} frames  —  baseline saved",
                    (14, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (180, 180, 180), 1, cv2.LINE_AA)

    # ── Metric bars (live, bottom panel) ─────────────────
    if metrics and phase in ("recording", "done"):
        panel_top = h_f - 170
        overlay   = frame.copy()
        cv2.rectangle(overlay, (0, panel_top - 8), (260, h_f), (12, 14, 18), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "LIVE METRICS", (12, panel_top + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 110, 120), 1)

        rows = [
            ("Head fwd",    metrics["head_forward"],   -0.05, 0.08,  (0, 200, 130)),
            ("Shoulder",    metrics["shoulder_asym"],   0.0,  0.08,  (30, 165, 255)),
            ("Spinal",      metrics["spinal_offset"],   0.0,  0.10,  (30, 165, 255)),
            ("Neck °",      metrics["neck_angle_deg"],  0.0,  30.0,  (0, 200, 130)),
            ("Torso °",     metrics["torso_lean_deg"],  0.0,  20.0,  (0, 200, 130)),
        ]
        for i, (lbl, val, vmin, vmax, col) in enumerate(rows):
            draw_bar(frame, 12, panel_top + 26 + i * 28, 230, 10,
                     val, vmin, vmax, col, lbl)

    # ── Posture cue (right side) ──────────────────────────
    if phase in ("countdown", "recording"):
        tips = [
            "Sit as you normally would when",
            "working comfortably — not forced",
            "perfect posture, just YOUR natural",
            "comfortable upright position.",
            "",
            "Avoid looking down or craning.",
            "Stay still and breathe normally.",
        ]
        rx = w_f - 310
        cv2.rectangle(frame, (rx - 10, 118), (w_f, 118 + len(tips) * 22 + 16),
                      (18, 20, 26), -1)
        cv2.putText(frame, "POSTURE CUE", (rx, 136),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 90, 100), 1)
        for i, t in enumerate(tips):
            cv2.putText(frame, t, (rx, 155 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (170, 175, 185), 1, cv2.LINE_AA)


# ── Main calibration flow ─────────────────────────────────
def run_calibration():
    cap  = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.70,
                        min_tracking_confidence=0.70)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("\n╔══════════════════════════════════════════════╗")
    print("║    PostureGuard  —  Personal Calibration     ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  Sit in YOUR natural comfortable posture.    ║")
    print("║  Not perfect — just how you normally sit.    ║")
    print("║                                              ║")
    print("║  Recording for 60 seconds.                   ║")
    print("║  Press  Q  to cancel at any time.            ║")
    print("╚══════════════════════════════════════════════╝\n")

    # ── Phase 1: countdown ────────────────────────────────
    print(f"Get into position — starting in {WARMUP}s...")
    t_countdown = time.time()
    while True:
        elapsed = time.time() - t_countdown
        if elapsed >= WARMUP:
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
        draw_hud(frame, "countdown", elapsed, WARMUP, 0, None, 0)
        cv2.imshow("PostureGuard — Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("Calibration cancelled.")
            return

    # ── Phase 2: record ───────────────────────────────────
    print("Recording... hold your position.")
    all_metrics = []
    t_record    = time.time()
    last_metrics = None

    while True:
        elapsed = time.time() - t_record
        if elapsed >= CALIB_DURATION:
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
            m = compute_metrics(results.pose_landmarks.landmark)
            all_metrics.append(m)
            last_metrics = m
        else:
            h_f, w_f = frame.shape[:2]
            cv2.putText(frame, "No person detected — stay in frame!",
                        (14, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 80, 255), 2, cv2.LINE_AA)

        # Stability = how consistent neck_angle has been (lower std = more stable)
        stability = 0.0
        if len(all_metrics) >= 10:
            recent = [m["neck_angle_deg"] for m in all_metrics[-30:]]
            std    = np.std(recent)
            stability = max(0.0, 100.0 - std * 20)   # rough 0-100

        draw_hud(frame, "recording", elapsed, CALIB_DURATION,
                 len(all_metrics), last_metrics, stability)
        cv2.imshow("PostureGuard — Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Calibration cancelled early.")
            break

    # ── Phase 3: compute & validate ───────────────────────
    n = len(all_metrics)
    print(f"\nCaptured {n} valid frames.")

    if n < MIN_FRAMES:
        print(f"ERROR: Only {n} frames — need at least {MIN_FRAMES}.")
        print("Make sure you stay visible in the camera and try again.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Compute mean and std for each metric
    keys = list(all_metrics[0].keys())
    baseline = {}
    print("\n── Calibration Results ──────────────────────────")
    for k in keys:
        vals     = np.array([m[k] for m in all_metrics])
        # Trim outliers (beyond 2 std) before averaging
        mean     = float(np.mean(vals))
        std      = float(np.std(vals))
        trimmed  = vals[np.abs(vals - mean) <= 2 * std]
        t_mean   = float(np.mean(trimmed))
        t_std    = float(np.std(trimmed))

        baseline[k] = {
            "mean": round(t_mean, 5),
            "std":  round(t_std,  5),
            # Tolerance bands: good = within 1.5σ, moderate = within 3σ, bad = beyond
            "warn_moderate": round(t_mean + 1.5 * t_std, 5),
            "warn_bad":      round(t_mean + 3.0 * t_std, 5),
        }
        unit = "°" if "deg" in k else ""
        print(f"  {k:<20}  mean={t_mean:+.4f}{unit}  std={t_std:.4f}{unit}")

    # Stability score
    neck_vals  = np.array([m["neck_angle_deg"] for m in all_metrics])
    stability_score = max(0, round(100 - float(np.std(neck_vals)) * 15, 1))

    # Save
    output = {
        "calibrated_at":  datetime.now().isoformat(),
        "frames_used":    n,
        "stability_score": stability_score,
        "metrics":        baseline,
        "notes": (
            "warn_moderate = 1.5σ above personal mean  "
            "(first gentle nudge threshold). "
            "warn_bad = 3.0σ above personal mean "
            "(bad-posture classification threshold). "
            "Load this in realtime_posture.py to personalise alerts."
        )
    }

    with open(BASELINE_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Stability score   :  {stability_score}%")
    print(f"  Frames used       :  {n}")
    if stability_score < 60:
        print("  NOTE: Stability is low — try to keep still during calibration.")
        print("        Consider re-running calibrate.py for better results.")

    # ── Phase 4: show summary on screen for 4 seconds ─────
    t_done = time.time()
    while time.time() - t_done < 4:
        ret, frame = cap.read()
        if not ret:
            break
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        draw_hud(frame, "done", 0, 0, n, last_metrics, stability_score)
        cv2.imshow("PostureGuard — Calibration", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Baseline saved  →  {BASELINE_FILE}")
    print("  Next step: run  realtime_posture.py  — it will load this file")
    print("  automatically and personalise your alert thresholds.\n")


# ── How to use the baseline in realtime_posture.py ────────
def print_integration_guide():
    guide = """
── How to integrate user_baseline.json in realtime_posture.py ──────────

Add near the top of realtime_posture.py, after imports:

    import json, os
    BASELINE_FILE = "user_baseline.json"
    baseline = None
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            baseline = json.load(f)["metrics"]
        print("Personal baseline loaded.")

Then in your metrics display / alert logic, replace hardcoded thresholds:

    # BEFORE (generic):
    if metrics["neck_angle_deg"] > 22:
        # moderate warning

    # AFTER (personalised):
    neck_warn = baseline["neck_angle_deg"]["warn_moderate"] if baseline else 15.0
    if metrics["neck_angle_deg"] > neck_warn:
        # moderate warning

Apply the same pattern for head_forward, shoulder_asym, spinal_offset,
and torso_lean_deg.  The warn_bad threshold replaces your old hard thresholds
for triggering the BAD alert banner.
──────────────────────────────────────────────────────────────────────────
"""
    print(guide)


if __name__ == "__main__":
    run_calibration()
    print_integration_guide()
