"""
PostureGuard  —  Real-time posture detection  (v3)
──────────────────────────────────────────────────
Changes vs v2:
  • Loads user_baseline.json → personalised thresholds  (FR8/FR9)
  • OpticalFlowPredictor    → predictive early warning  (FR5/FR6)
  • Fatigue countdown in UI                              (FR7)
  • S key = snooze all alerts 5 min                     (FR11)
  • Issue-specific fix tip shown in bad-alert banner     (FR13)
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time, json, os, sys
from datetime import datetime

from optical_flow import OpticalFlowPredictor   # our new module

# ── Cross-platform beep ───────────────────────────────────
def play_alert():
    try:
        if sys.platform == "win32":
            import winsound; winsound.Beep(880, 400)
        else:
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=44100)
                sr  = 44100
                t_  = np.linspace(0, 0.4, int(sr * 0.4), False)
                wav = (np.sin(2 * np.pi * 880 * t_) * 28000).astype(np.int16)
                pygame.sndarray.make_sound(np.column_stack([wav, wav])).play()
            except Exception:
                print("\a", end="", flush=True)
    except Exception:
        pass

# ── LSTM model ────────────────────────────────────────────
class PostureLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128,
                 num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(0.4)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.norm(out[:, -1, :])))

model = PostureLSTM()
model.load_state_dict(torch.load("posture_model.pth", map_location="cpu"))
model.eval()
print("LSTM model loaded.")

# ── Load personal baseline (FR8/FR9) ─────────────────────
BASELINE_FILE = "user_baseline.json"
baseline = None
if os.path.exists(BASELINE_FILE):
    with open(BASELINE_FILE) as f:
        baseline = json.load(f).get("metrics", None)
    print(f"Personal baseline loaded  ({BASELINE_FILE})")
else:
    print("No baseline found — using generic thresholds.")
    print("Run calibrate.py first for personalised alerts.")

def thr(metric, level, fallback):
    """Return personalised threshold, or fallback if no baseline."""
    if baseline and metric in baseline:
        return baseline[metric][level]
    return fallback

THRESH = {
    "head_forward":   {"moderate": thr("head_forward",   "warn_moderate", 0.030),
                       "bad":      thr("head_forward",   "warn_bad",      0.060)},
    "shoulder_asym":  {"moderate": thr("shoulder_asym",  "warn_moderate", 0.030),
                       "bad":      thr("shoulder_asym",  "warn_bad",      0.060)},
    "spinal_offset":  {"moderate": thr("spinal_offset",  "warn_moderate", 0.040),
                       "bad":      thr("spinal_offset",  "warn_bad",      0.080)},
    "neck_angle_deg": {"moderate": thr("neck_angle_deg", "warn_moderate", 15.0),
                       "bad":      thr("neck_angle_deg", "warn_bad",      25.0)},
    "torso_lean_deg": {"moderate": thr("torso_lean_deg", "warn_moderate", 10.0),
                       "bad":      thr("torso_lean_deg", "warn_bad",      18.0)},
}

# ── Corrective tips per metric (FR13) ─────────────────────
TIPS = {
    "head_forward":   "Tuck chin back — ears over shoulders",
    "shoulder_asym":  "Level your shoulders — roll them back",
    "spinal_offset":  "Centre your spine — sit over sit bones",
    "neck_angle_deg": "Raise your gaze — lift screen to eye level",
    "torso_lean_deg": "Straighten back — press lumbar into chair",
}

def worst_metric(m):
    best_k, best_r = "neck_angle_deg", 0.0
    for k, v in m.items():
        r = abs(v) / (THRESH.get(k, {}).get("bad", 1.0) + 1e-6)
        if r > best_r:
            best_r, best_k = r, k
    return best_k

# ── Optical flow predictor (FR5/FR6/FR7) ──────────────────
predictor = OpticalFlowPredictor(fps=20.0, window_sec=30.0)

# ── MediaPipe + webcam ────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose    = mp_pose.Pose(min_detection_confidence=0.65,
                       min_tracking_confidence=0.65)
cap     = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("ERROR: webcam not accessible.")

SEQ_LEN      = 30
frame_buffer = deque(maxlen=SEQ_LEN)

CLS = {
    0: {"name": "GOOD POSTURE",     "color": (50, 220, 130), "short": "good"},
    1: {"name": "MODERATE POSTURE", "color": (30, 165, 255), "short": "moderate"},
    2: {"name": "BAD POSTURE",      "color": (60,  60, 230), "short": "bad"},
}

# ── Session log ───────────────────────────────────────────
SESSION_DIR  = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
sid          = datetime.now().strftime("%Y%m%d_%H%M%S")
session_file = os.path.join(SESSION_DIR, f"session_{sid}.json")
session = {
    "session_id": sid, "start_time": datetime.now().isoformat(),
    "baseline_used": baseline is not None,
    "frames": [], "alerts": [], "summary": {},
}

def log_frame(t, cls, conf, mets, flow_mean):
    session["frames"].append({
        "t": round(t, 2), "class": int(cls),
        "cls_name": CLS[cls]["short"],
        "conf": round(float(conf), 3),
        "flow_mean": round(float(flow_mean), 5),
        **{k: round(float(v), 4) for k, v in mets.items()},
    })

def finalise():
    session["end_time"] = datetime.now().isoformat()
    frames = session["frames"]
    if frames:
        n = len(frames)
        counts = {v["short"]: 0 for v in CLS.values()}
        for f in frames:
            counts[f["cls_name"]] = counts.get(f["cls_name"], 0) + 1
        dur = frames[-1]["t"] - frames[0]["t"] if n > 1 else 0
        session["summary"] = {
            "duration_s":   round(dur, 1), "total_frames": n,
            "good_pct":     round(100 * counts["good"]     / n, 1),
            "moderate_pct": round(100 * counts["moderate"] / n, 1),
            "bad_pct":      round(100 * counts["bad"]      / n, 1),
            "total_alerts": len(session["alerts"]),
        }
    with open(session_file, "w") as f:
        json.dump(session, f, indent=2)
    print(f"\nSession saved  →  {session_file}")
    s = session["summary"]
    if s:
        print(f"  Good {s['good_pct']}%  Moderate {s['moderate_pct']}%  "
              f"Bad {s['bad_pct']}%  Alerts {s['total_alerts']}")

# ── Metrics ───────────────────────────────────────────────
def compute_metrics(lms):
    def pt(i): return np.array([lms[i].x, lms[i].y, lms[i].z])
    ear  = (pt(7)  + pt(8))  / 2
    sh   = (pt(11) + pt(12)) / 2
    hip  = (pt(23) + pt(24)) / 2
    l_sh, r_sh = pt(11), pt(12)
    hf  = float(sh[0] - ear[0])
    sa  = float(abs(l_sh[1] - r_sh[1]))
    so  = float(abs(sh[0] - hip[0]))
    vn  = sh[:2] - ear[:2]
    na  = float(np.degrees(np.arctan2(abs(vn[0]), abs(vn[1]) + 1e-6)))
    vt  = hip[:2] - sh[:2]
    tl  = float(np.degrees(np.arctan2(abs(vt[0]), abs(vt[1]) + 1e-6)))
    return {"head_forward": hf, "shoulder_asym": sa, "spinal_offset": so,
            "neck_angle_deg": na, "torso_lean_deg": tl}

# ── State ─────────────────────────────────────────────────
current_cls    = -1
label_text     = "Warming up..."
label_color    = (200, 200, 60)
confidence     = 0.0
metrics        = {}
bad_start      = None
moderate_start = None
last_beep      = 0.0
snooze_until   = 0.0

BAD_THRESHOLD      = 5
MODERATE_THRESHOLD = 15
BEEP_COOLDOWN      = 10
t0 = time.time()

print("Running  —  Q = quit   S = snooze 5 min\n")

# ── Main loop ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now  = time.time()
    t    = now - t0
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    detected = res.pose_landmarks is not None

    if detected:
        mp_draw.draw_landmarks(
            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 240, 160), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0, 180, 120), thickness=2),
        )
        kp = [v for lm in res.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
        frame_buffer.append(kp)
        metrics = compute_metrics(res.pose_landmarks.landmark)

        if len(frame_buffer) == SEQ_LEN:
            tensor = torch.from_numpy(
                np.array(frame_buffer, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1).squeeze().tolist()
            current_cls = int(np.argmax(probs))
            confidence  = probs[current_cls]
            label_text  = CLS[current_cls]["name"]
            label_color = CLS[current_cls]["color"]

            if current_cls == 2:
                bad_start      = bad_start or now
                moderate_start = None
            elif current_cls == 1:
                moderate_start = moderate_start or now
                bad_start      = None
            else:
                bad_start = moderate_start = None

            log_frame(t, current_cls, confidence, metrics,
                      predictor.get_debug_info()["flow_mean"])
    else:
        label_text  = "No person detected"
        label_color = (100, 100, 100)
        frame_buffer.clear()
        bad_start = moderate_start = None
        cv2.putText(frame, "Return to camera view",
                    (w // 2 - 170, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255), 2)

    # ── Optical flow update ───────────────────────────────
    try:
        of_warning, fatigue_msg = predictor.update(frame, current_cls)
    except Exception:
        of_warning, fatigue_msg = None, None
    flow_info = predictor.get_debug_info()

    # ── Alert state ───────────────────────────────────────
    snoozed   = now < snooze_until
    bad_dur   = (now - bad_start)      if bad_start      else 0.0
    mod_dur   = (now - moderate_start) if moderate_start else 0.0
    bad_alert = bad_dur  >= BAD_THRESHOLD      and not snoozed
    mod_alert = mod_dur  >= MODERATE_THRESHOLD and not snoozed

    if bad_alert and (now - last_beep) > BEEP_COOLDOWN:
        play_alert()
        last_beep = now
        session["alerts"].append({
            "t": round(t, 2), "type": "bad",
            "bad_duration": round(bad_dur, 1),
            "tip": TIPS.get(worst_metric(metrics), ""),
        })

    # ── Draw UI ───────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 118), (15, 15, 20), -1)
    cv2.putText(frame, label_text, (14, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, label_color, 2, cv2.LINE_AA)

    if current_cls >= 0 and len(frame_buffer) == SEQ_LEN:
        bmax = w - 28
        cv2.rectangle(frame, (14, 62), (14 + bmax, 71), (45, 45, 45), -1)
        cv2.rectangle(frame, (14, 62),
                      (14 + int(confidence * bmax), 71), label_color, -1)
        cv2.putText(frame, f"Conf {confidence*100:.0f}%",
                    (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1)
    else:
        cv2.putText(frame, f"Buffer {len(frame_buffer)}/{SEQ_LEN}",
                    (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 60), 1)

    mm, ss = divmod(int(t), 60)
    snooze_txt = "  SNOOZED" if snoozed else ""
    cv2.putText(frame, f"{mm:02d}:{ss:02d}{snooze_txt}",
                (w - 185, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (200, 130, 30) if snoozed else (155, 155, 155), 1)
    if bad_start:
        cv2.putText(frame, f"Bad {bad_dur:.0f}s/{BAD_THRESHOLD}s",
                    (w - 175, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (60, 60, 230), 1, cv2.LINE_AA)

    # Alert banners — stack from y=120 downward
    banner_bottom = 120
    if bad_alert:
        tip = TIPS.get(worst_metric(metrics), "Fix your posture")
        cv2.rectangle(frame, (0, 120), (w, 188), (40, 0, 160), -1)
        cv2.putText(frame, "!  FIX YOUR POSTURE  !",
                    (w // 2 - 180, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        cv2.putText(frame, tip,
                    (w // 2 - min(len(tip) * 4, 280), 174),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 200, 255), 1)
        banner_bottom = 188
    elif mod_alert:
        cv2.rectangle(frame, (0, 120), (w, 175), (20, 90, 140), -1)
        cv2.putText(frame, "Posture drifting — sit straight",
                    (w // 2 - 190, 154),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 230, 180), 1)
        banner_bottom = 175

    if of_warning and not snoozed:
        cv2.rectangle(frame, (0, banner_bottom), (w, banner_bottom + 44),
                      (20, 70, 90), -1)
        cv2.putText(frame, f"  PREDICT: {of_warning}",
                    (10, banner_bottom + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 230, 255), 1)

    if fatigue_msg:
        cv2.putText(frame, fatigue_msg,
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 100), 1)

    if metrics and detected:
        py = h - 145
        cv2.rectangle(frame, (0, py - 8), (250, h), (14, 14, 20), -1)
        cv2.putText(frame, "METRICS", (12, py + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 90, 100), 1)
        rows = [("Head fwd",  "head_forward",   ""),
                ("Shoulder",  "shoulder_asym",  ""),
                ("Spinal",    "spinal_offset",  ""),
                ("Neck",      "neck_angle_deg", "°"),
                ("Torso",     "torso_lean_deg", "°")]
        for i, (lbl, key, unit) in enumerate(rows):
            val = metrics[key]
            c   = ((50, 220, 130) if abs(val) < THRESH[key]["moderate"]
                   else (30, 165, 255) if abs(val) < THRESH[key]["bad"]
                   else (60, 60, 230))
            cv2.putText(frame, f"{lbl:<10} {val:+.3f}{unit}",
                        (12, py + 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, c, 1, cv2.LINE_AA)

    cv2.putText(frame,
                f"flow {flow_info['flow_mean']:.4f}  "
                f"bad {flow_info['bad_rate_pct']:.0f}%",
                (w - 240, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (55, 65, 75), 1)

    cv2.imshow("PostureGuard — Live Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        snooze_until = time.time() + 300
        print("Alerts snoozed 5 min.")

cap.release()
cv2.destroyAllWindows()
finalise()
print("PostureGuard stopped.")
