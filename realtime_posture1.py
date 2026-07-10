"""
PostureGuard  —  Real-time posture detection
3-class LSTM  (Good / Moderate / Bad)
+ biomechanical metrics  + session JSON logging  + cross-platform audio
Covers FR2, FR3, FR4, FR11, FR14
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time, json, os, sys
from datetime import datetime

# ── Cross-platform alert sound ────────────────────────────
def play_alert():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(880, 400)
        else:
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=44100)
                sr = 44100
                t  = np.linspace(0, 0.4, int(sr * 0.4), False)
                wave = (np.sin(2 * np.pi * 880 * t) * 28000).astype(np.int16)
                stereo = np.column_stack([wave, wave])
                pygame.sndarray.make_sound(stereo).play()
            except Exception:
                print("\a", end="", flush=True)   # terminal bell fallback
    except Exception:
        pass


# ── 1. LSTM model — must match train_lstm.py ─────────────
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
        out = self.norm(out[:, -1, :])
        return self.fc(self.drop(out))


# ── 2. Load model ─────────────────────────────────────────
model = PostureLSTM()
model.load_state_dict(torch.load("posture_model.pth", map_location="cpu"))
model.eval()
print("Model loaded  —  3-class  (Good / Moderate / Bad)")

# ── 3. MediaPipe + webcam ─────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose    = mp_pose.Pose(min_detection_confidence=0.65,
                       min_tracking_confidence=0.65)
cap     = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("ERROR: Could not open webcam.")

SEQ_LEN      = 30
frame_buffer = deque(maxlen=SEQ_LEN)

# ── 4. Class config ───────────────────────────────────────
CLS = {
    0: {"name": "GOOD POSTURE",     "color": (50,  220, 130), "short": "good"},
    1: {"name": "MODERATE POSTURE", "color": (30,  165, 255), "short": "moderate"},
    2: {"name": "BAD POSTURE",      "color": (60,   60, 230), "short": "bad"},
}

# ── 5. Session logging ────────────────────────────────────
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
sid          = datetime.now().strftime("%Y%m%d_%H%M%S")
session_file = os.path.join(SESSION_DIR, f"session_{sid}.json")

session = {
    "session_id":  sid,
    "start_time":  datetime.now().isoformat(),
    "end_time":    None,
    "frames":      [],   # {t, cls, cls_name, conf, metrics…}
    "alerts":      [],   # {t, type, bad_duration}
    "summary":     {},
}

def log_frame(t, cls, conf, metrics):
    session["frames"].append({
        "t":         round(t, 2),
        "class":     int(cls),
        "cls_name":  CLS[cls]["short"],
        "conf":      round(float(conf), 3),
        **{k: round(float(v), 4) for k, v in metrics.items()},
    })

def finalise_session():
    session["end_time"] = datetime.now().isoformat()
    frames = session["frames"]
    if frames:
        counts = {v["short"]: 0 for v in CLS.values()}
        for f in frames:
            counts[f["cls_name"]] += 1
        n = len(frames)
        dur = frames[-1]["t"] - frames[0]["t"] if n > 1 else 0
        session["summary"] = {
            "duration_s":   round(dur, 1),
            "total_frames": n,
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
        print(f"  Duration   : {s['duration_s']}s")
        print(f"  Good       : {s['good_pct']}%")
        print(f"  Moderate   : {s['moderate_pct']}%")
        print(f"  Bad        : {s['bad_pct']}%")
        print(f"  Alerts     : {s['total_alerts']}")


# ── 6. Biomechanical metrics ─────────────────────────────
#
# MediaPipe landmark indices (relevant ones):
#   0=nose  7=l_ear  8=r_ear
#   11=l_shoulder  12=r_shoulder
#   23=l_hip       24=r_hip
#
# All coords are normalised 0–1 (x right, y down in image).
# Metrics are camera-relative but still strongly correlate with
# real posture because keypoint proportions are invariant.

def compute_metrics(lms):
    def pt(i):
        return np.array([lms[i].x, lms[i].y, lms[i].z])

    nose       = pt(0)
    ear_mid    = (pt(7)  + pt(8))  / 2
    sh_mid     = (pt(11) + pt(12)) / 2
    hip_mid    = (pt(23) + pt(24)) / 2
    l_sh, r_sh = pt(11), pt(12)

    # Head-forward  : ear midpoint ahead of shoulder midpoint in x
    # Positive = forward head posture (bad)
    head_forward = float(sh_mid[0] - ear_mid[0])

    # Shoulder asymmetry  : vertical difference (y increases downward)
    shoulder_asym = float(abs(l_sh[1] - r_sh[1]))

    # Spinal lateral offset  : horizontal drift shoulder vs hip midpoint
    spinal_offset = float(abs(sh_mid[0] - hip_mid[0]))

    # Neck angle  : angle of ear→shoulder vector from vertical (degrees)
    vec = sh_mid[:2] - ear_mid[:2]
    neck_angle = float(np.degrees(np.arctan2(abs(vec[0]),
                                              abs(vec[1]) + 1e-6)))

    # Torso lean  : angle of shoulder→hip from vertical
    tvec = hip_mid[:2] - sh_mid[:2]
    torso_lean = float(np.degrees(np.arctan2(abs(tvec[0]),
                                              abs(tvec[1]) + 1e-6)))

    return {
        "head_forward":    head_forward,
        "shoulder_asym":   shoulder_asym,
        "spinal_offset":   spinal_offset,
        "neck_angle_deg":  neck_angle,
        "torso_lean_deg":  torso_lean,
    }


# ── 7. Alert state ────────────────────────────────────────
current_cls    = -1
label_text     = "Warming up..."
label_color    = (200, 200, 60)
confidence     = 0.0
metrics        = {}

bad_start      = None
moderate_start = None
last_beep      = 0.0

BAD_THRESHOLD      = 5     # seconds of bad before alert fires
MODERATE_THRESHOLD = 15    # seconds of moderate before gentle nudge
BEEP_COOLDOWN      = 10    # seconds between repeated beeps

t0 = time.time()
print("PostureGuard running  —  press  Q  to quit\n")


# ── Main loop ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam read failed — exiting.")
        break

    now = time.time()
    t   = now - t0
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    h, w = frame.shape[:2]

    detected = res.pose_landmarks is not None

    # ── Pose detection ────────────────────────────────────
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
            seq    = np.array(frame_buffer, dtype=np.float32)
            tensor = torch.from_numpy(seq).unsqueeze(0)

            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1).squeeze().tolist()

            current_cls = int(np.argmax(probs))
            confidence  = probs[current_cls]
            label_text  = CLS[current_cls]["name"]
            label_color = CLS[current_cls]["color"]

            # Reset/start timers
            if current_cls == 2:          # bad
                if bad_start is None:
                    bad_start = now
                moderate_start = None
            elif current_cls == 1:        # moderate
                bad_start = None
                if moderate_start is None:
                    moderate_start = now
            else:                         # good
                bad_start = moderate_start = None

            log_frame(t, current_cls, confidence, metrics)
    else:
        # FR2: show status when no person detected
        label_text  = "No person detected"
        label_color = (100, 100, 100)
        frame_buffer.clear()
        bad_start = moderate_start = None
        cv2.putText(frame, "Return to camera view",
                    (w // 2 - 170, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255), 2, cv2.LINE_AA)

    # ── Alert logic (FR11) ────────────────────────────────
    bad_dur  = (now - bad_start)      if bad_start      else 0.0
    mod_dur  = (now - moderate_start) if moderate_start else 0.0
    bad_alert = bad_dur  >= BAD_THRESHOLD
    mod_alert = mod_dur  >= MODERATE_THRESHOLD

    if bad_alert and (now - last_beep) > BEEP_COOLDOWN:
        play_alert()
        last_beep = now
        session["alerts"].append({
            "t":            round(t, 2),
            "type":         "bad",
            "bad_duration": round(bad_dur, 1),
        })

    # ── Draw UI ───────────────────────────────────────────
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 118), (15, 15, 20), -1)

    # Class label
    cv2.putText(frame, label_text, (14, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, label_color, 2, cv2.LINE_AA)

    # Confidence bar
    if current_cls >= 0 and len(frame_buffer) == SEQ_LEN:
        bx, by, bh2 = 14, 62, 9
        bmax = w - 28
        cv2.rectangle(frame, (bx, by), (bx + bmax, by + bh2), (45, 45, 45), -1)
        cv2.rectangle(frame, (bx, by),
                      (bx + int(confidence * bmax), by + bh2), label_color, -1)
        cv2.putText(frame, f"Confidence  {confidence*100:.0f}%",
                    (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (190, 190, 190), 1)
    elif len(frame_buffer) < SEQ_LEN:
        cv2.putText(frame, f"Buffer  {len(frame_buffer)}/{SEQ_LEN}",
                    (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 180, 60), 1)

    # Session clock (top-right)
    mm, ss = divmod(int(t), 60)
    cv2.putText(frame, f"{mm:02d}:{ss:02d}",
                (w - 90, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (155, 155, 155), 1)

    # Bad-posture timer
    if bad_start:
        cv2.putText(frame, f"Bad  {bad_dur:.0f}s / {BAD_THRESHOLD}s",
                    (w - 175, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (60, 60, 230), 1, cv2.LINE_AA)

    # Alert banners
    if bad_alert:
        cv2.rectangle(frame, (0, 120), (w, 180), (40, 0, 160), -1)
        cv2.putText(frame, "!  FIX YOUR POSTURE NOW  !",
                    (w // 2 - 200, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    elif mod_alert:
        cv2.rectangle(frame, (0, 120), (w, 180), (20, 90, 140), -1)
        cv2.putText(frame, "Posture drifting — try sitting straight",
                    (w // 2 - 220, 158),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 230, 180), 1, cv2.LINE_AA)

    # Metrics panel (bottom-left) — FR3: additional postural metrics
    if metrics and detected:
        panel_y = h - 135
        cv2.rectangle(frame, (0, panel_y - 8), (240, h), (14, 14, 20), -1)

        def metric_color(val, good_thresh, bad_thresh):
            if abs(val) < good_thresh:
                return (50, 220, 130)
            if abs(val) < bad_thresh:
                return (30, 165, 255)
            return (60, 60, 230)

        rows = [
            ("Head forward",   metrics["head_forward"],   0.02, 0.05),
            ("Shoulder diff",  metrics["shoulder_asym"],  0.03, 0.06),
            ("Spinal offset",  metrics["spinal_offset"],  0.04, 0.08),
            ("Neck angle",     metrics["neck_angle_deg"], 12.0, 22.0),
            ("Torso lean",     metrics["torso_lean_deg"], 8.0,  16.0),
        ]
        cv2.putText(frame, "METRICS", (12, panel_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 110), 1)
        for i, (label, val, gt, bt) in enumerate(rows):
            unit   = "°" if "angle" in label.lower() or "lean" in label.lower() else ""
            color  = metric_color(val, gt, bt)
            cv2.putText(frame, f"{label:<14} {val:+.3f}{unit}",
                        (12, panel_y + 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    cv2.imshow("PostureGuard — Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── Cleanup ───────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
finalise_session()
print("PostureGuard stopped.")
