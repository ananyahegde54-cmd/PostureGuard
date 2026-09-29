"""
PostureGuard  —  Live Monitor server (wraps realtime_posture.py logic)
──────────────────────────────────────────────────────────────────────
Run:   python app.py
Open:  http://localhost:5000      (Chrome/Edge/Firefox)

Place next to: realtime_posture.py, optical_flow.py,
               posture_model.pth, database.py
Requires:      pip install flask
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time, json, os, sys, threading, hashlib, uuid
from datetime import datetime
from functools import wraps
from flask import (Flask, Response, jsonify, request, render_template,
                    session, redirect, url_for)

from optical_flow import OpticalFlowPredictor
import database

# ── Database-backed user store (SQLite: postureguard.db) ───
database.init_db()

def hash_pw(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ── Login validation ─────────────────────────────────────
# No password/username format rules. Any non-empty credentials are accepted.

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login_page"))
        return view(*args, **kwargs)
    return wrapped

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

# ── Personal baseline (FR8/FR9) ───────────────────────────
# Baselines are stored in SQLite per user.  We never use one shared
# user_baseline.json for all accounts.
baseline = None

def build_thresholds(active_baseline):
    def value(metric, level, fallback):
        if active_baseline and metric in active_baseline:
            return active_baseline[metric].get(level, fallback)
        return fallback

    return {
        "head_forward":   {"moderate": value("head_forward",   "warn_moderate", 0.030),
                           "bad":      value("head_forward",   "warn_bad",      0.060)},
        "shoulder_asym":  {"moderate": value("shoulder_asym",  "warn_moderate", 0.030),
                           "bad":      value("shoulder_asym",  "warn_bad",      0.060)},
        "spinal_offset":  {"moderate": value("spinal_offset",  "warn_moderate", 0.040),
                           "bad":      value("spinal_offset",  "warn_bad",      0.080)},
        "neck_angle_deg": {"moderate": value("neck_angle_deg", "warn_moderate", 15.0),
                           "bad":      value("neck_angle_deg", "warn_bad",      25.0)},
        "torso_lean_deg": {"moderate": value("torso_lean_deg", "warn_moderate", 10.0),
                           "bad":      value("torso_lean_deg", "warn_bad",      18.0)},
    }

THRESH = build_thresholds(None)

def load_user_baseline(username):
    row = database.get_baseline(username)
    return row["metrics"] if row and row.get("metrics") else None

def set_active_user(username):
    """Load only this user's baseline into the live detector."""
    global baseline, THRESH
    baseline = load_user_baseline(username)
    THRESH = build_thresholds(baseline)
    with lock:
        STATE["thresh"] = THRESH
        STATE["baseline_used"] = baseline is not None
    print(f"Active user: {username} | user baseline: {'loaded' if baseline else 'not found'}")

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

# ── Posture tips (were missing -> NameError killed the camera thread on the first bad-posture alert) ──
TIPS = {
    "head_forward":   "Tuck chin back — ears over shoulders",
    "shoulder_asym":  "Level your shoulders — roll them back",
    "spinal_offset":  "Centre your spine — sit over sit bones",
    "neck_angle_deg": "Raise your gaze — lift screen to eye level",
    "torso_lean_deg": "Straighten back — press lumbar into chair",
}
def worst_metric(m):
    best_k, best_r = "neck_angle_deg", 0.0
    for k, v in (m or {}).items():
        r = abs(v) / (THRESH.get(k, {}).get("bad", 1.0) + 1e-6)
        if r > best_r:
            best_r, best_k = r, k
    return best_k

CLS = {
    0: {"name": "GOOD POSTURE",     "color": (50, 220, 130), "short": "good"},
    1: {"name": "MODERATE POSTURE", "color": (30, 165, 255), "short": "moderate"},
    2: {"name": "BAD POSTURE",      "color": (60,  60, 230), "short": "bad"},
}

# ── Shared state (thread-safe) ────────────────────────────
lock        = threading.Lock()
STATE       = {
    "running": False, "detected": False, "buffer": 0, "seq_len": 30,
    "cls": -1, "cls_name": "Warming up…", "conf": 0.0,
    "metrics": {}, "thresh": THRESH,
    "bad_dur": 0.0, "mod_dur": 0.0, "t": 0.0,
    "bad_alert": False, "mod_alert": False,
    "of_warning": None, "fatigue": None,
    "snoozed": False, "snooze_left_s": 0,
    "flow_mean": 0.0, "bad_rate_pct": 0.0,
    "alerts": [], "baseline_used": baseline is not None,
    "summary": None, "camera_error": None, "calibrating": False,
    # wellness coach
    "good_dur": 0.0, "sit_dur": 0.0, "wellness_due": False,
    "wellness_after_s": 1800, "breaks_taken": 0, "focus_metric": None,
}
LATEST_JPEG      = None
stop_event       = threading.Event()
SNOOZE_UNTIL     = [0.0]          # shared: UI snooze button writes here
# ── Wellness coach settings ───────────────────────────────
# After this many seconds of continuous GOOD posture the robot suggests a
# movement break (research: break up sitting at least every ~30 min).
# For quick testing run:  set WELLNESS_GOOD_SECS=60   (Windows)  before  python app.py
WELLNESS_GOOD_SECS   = int(os.environ.get("WELLNESS_GOOD_SECS", 30 * 60))
GOOD_STREAK_GRACE_S  = 45     # a short slouch (<45 s) does not reset the good-posture streak
AWAY_RESET_S         = 60     # leaving the frame for 60 s counts as a break (streak resets)
WELLNESS_CTRL    = {}             # commands from /api/wellness/ack -> read by detection loop
detection_thread = None           # tracks the current camera/detection thread
calibration_mode = False          # True only while the user is recording a baseline
active_username = None            # current logged-in user used by the camera thread

def stop_calibration_mode():
    global calibration_mode
    with lock:
        calibration_mode = False
        STATE["calibrating"] = False
        STATE["cls"] = -1
        STATE["cls_name"] = "Warming up…"
        STATE["conf"] = 0.0
    frame_buffer.clear()

def stop_camera():
    """Stop the webcam/detection thread completely. Used on logout/login so
    the camera cannot keep running or beep while the user is on the login page."""
    global detection_thread, LATEST_JPEG
    stop_event.set()
    thread = detection_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
    detection_thread = None
    with lock:
        STATE["running"] = False
        STATE["calibrating"] = False
        STATE["bad_alert"] = False
        STATE["mod_alert"] = False
        STATE["cls"] = -1
        STATE["cls_name"] = "Camera stopped"
        STATE["conf"] = 0.0
        STATE["camera_error"] = None
        LATEST_JPEG = None

def ensure_camera_running():
    """(Re)starts the camera/detection loop if it isn't already running.
    This is what lets the camera turn back on after 'End session' was
    pressed — otherwise it only ever ran once, at server startup."""
    global detection_thread
    if detection_thread is None or not detection_thread.is_alive():
        stop_event.clear()
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()

# ── Session log ───────────────────────────────────────────
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
posture_session_file = None
posture_session = None

def log_frame(t, cls, conf, mets, flow_mean):
    posture_session["frames"].append({
        "t": round(t, 2), "class": int(cls),
        "cls_name": CLS[cls]["short"],
        "conf": round(float(conf), 3),
        "flow_mean": round(float(flow_mean), 5),
        **{k: round(float(v), 4) for k, v in mets.items()},
    })

def finalise():
    global posture_session_file
    posture_session["end_time"] = datetime.now().isoformat()
    frames = posture_session["frames"]
    if frames:
        n = len(frames)
        counts = {v["short"]: 0 for v in CLS.values()}
        for f in frames:
            counts[f["cls_name"]] = counts.get(f["cls_name"], 0) + 1
        dur = frames[-1]["t"] - frames[0]["t"] if n > 1 else 0
        posture_session["summary"] = {
            "duration_s": round(dur, 1), "total_frames": n,
            "good_pct": round(100 * counts["good"] / n, 1),
            "moderate_pct": round(100 * counts["moderate"] / n, 1),
            "bad_pct": round(100 * counts["bad"] / n, 1),
            "total_alerts": len(posture_session["alerts"]),
        }

    posture_session.setdefault("summary", {})
    if posture_session["summary"] is not None:
        posture_session["summary"]["breaks_taken"] = len(posture_session.get("breaks", []))

    # SQLite is now the source of truth for dashboard history.
    if active_username:
        database.save_session(active_username, posture_session)
        print(f"\nSession saved to database  →  user={active_username}, session={posture_session['session_id']}")

    # Keep the JSON file as a local backup/debug record. The dashboard does NOT
    # read these files, so old shared JSON sessions cannot leak into a user.
    with open(posture_session_file, "w") as f:
        json.dump(posture_session, f, indent=2)
    print(f"Local session backup → {posture_session_file}")
    return posture_session["summary"]

# ── Detection thread (realtime_posture.py main loop) ──────
predictor      = OpticalFlowPredictor(fps=20.0, window_sec=30.0)
SEQ_LEN        = 30
frame_buffer   = deque(maxlen=SEQ_LEN)
BAD_THRESHOLD, MODERATE_THRESHOLD, BEEP_COOLDOWN = 5, 15, 10

def detection_loop():
    global LATEST_JPEG, posture_session, posture_session_file, calibration_mode

    # fresh session-log file every time the camera (re)starts
    username_for_session = active_username
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_part = uuid.uuid4().hex[:8]
    sid = f"{timestamp}_{unique_part}"
    posture_session_file = os.path.join(SESSION_DIR, f"session_{sid}.json")
    posture_session      = {
        "session_id": sid, "username": username_for_session,
        "start_time": datetime.now().isoformat(),
        "baseline_used": baseline is not None,
        "frames": [], "alerts": [], "breaks": [], "summary": {},
    }

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose    = mp_pose.Pose(min_detection_confidence=0.65,
                           min_tracking_confidence=0.65)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
    if not cap.isOpened():
        with lock:
            STATE["camera_error"] = "Webcam not accessible"
        print("CAMERA: cv2.VideoCapture could not open the device at all.")
        return

    print("CAMERA: opened successfully, warming up...")
    for i in range(10):
        ok_warmup, _ = cap.read()
        print(f"   warm-up read {i+1}/10 -> success={ok_warmup}")
        time.sleep(0.1)

    current_cls, label_text = -1, "Warming up..."
    label_color, confidence = (200, 200, 60), 0.0
    metrics, bad_start, moderate_start = {}, None, None
    last_beep = 0.0
    t0 = time.time()
    frame_buffer.clear()

    # wellness-coach trackers (local to this session)
    good_start = nongood_start = sit_start = away_start = None
    wellness_next_at = 0.0
    worst_counts = {}
    good_dur = sit_dur = 0.0
    wellness_due = False

    with lock:
        STATE["running"] = True
        STATE["camera_error"] = None
        STATE["summary"] = None

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("CAMERA: cap.read() failed mid-loop - camera disconnected or grabbed by another app.")
            break
        try:
            now, t = time.time(), time.time() - t0
            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = pose.process(rgb)
            detected = res.pose_landmarks is not None

            # Calibration uses the same MediaPipe measurements, but it MUST NOT
            # run the LSTM classifier or draw Good/Moderate/Bad overlays.
            # The user is simply sitting naturally while we learn their baseline.
            with lock:
                is_calibrating = calibration_mode

            if detected:
                kp = [v for lm in res.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
                frame_buffer.append(kp)
                metrics = compute_metrics(res.pose_landmarks.landmark)

                if is_calibrating:
                    current_cls = -1
                    confidence = 0.0
                    label_text = "CALIBRATING"
                    label_color = (200, 200, 200)
                    bad_start = moderate_start = None
                else:
                    mp_draw.draw_landmarks(
                        frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 240, 160), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(0, 180, 120), thickness=2),
                    )
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
                            bad_start, moderate_start = bad_start or now, None
                        elif current_cls == 1:
                            moderate_start, bad_start = moderate_start or now, None
                        else:
                            bad_start = moderate_start = None
                        log_frame(t, current_cls, confidence, metrics,
                                  predictor.get_debug_info()["flow_mean"])
            else:
                label_text, label_color = ("CALIBRATING — no person detected", (180, 180, 180)) if is_calibrating else ("No person detected", (100, 100, 100))
                frame_buffer.clear()
                bad_start = moderate_start = None

            if is_calibrating:
                # No optical-flow warning, no beep, no posture alerts during baseline capture.
                of_warning, fatigue_msg = None, None
                flow_info = predictor.get_debug_info()
                snoozed = False
                bad_dur = mod_dur = 0.0
                bad_alert = mod_alert = False
            else:
                try:
                    of_warning, fatigue_msg = predictor.update(frame, current_cls)
                except Exception:
                    of_warning, fatigue_msg = None, None
                flow_info = predictor.get_debug_info()

                snoozed   = now < SNOOZE_UNTIL[0]
                bad_dur   = (now - bad_start)      if bad_start      else 0.0
                mod_dur   = (now - moderate_start) if moderate_start else 0.0
                bad_alert = bad_dur >= BAD_THRESHOLD and not snoozed
                mod_alert = mod_dur >= MODERATE_THRESHOLD and not snoozed

                if bad_alert and (now - last_beep) > BEEP_COOLDOWN:
                    threading.Thread(target=play_alert, daemon=True).start()
                    last_beep = now
                    tip = TIPS.get(worst_metric(metrics), "")
                    posture_session["alerts"].append({
                        "t": round(t, 2), "type": "bad",
                        "bad_duration": round(bad_dur, 1), "tip": tip,
                    })

            # ── Wellness coach: streak tracking + break commands ──────────
            with lock:
                _cmd   = WELLNESS_CTRL.pop("cmd", None)
                _kind  = WELLNESS_CTRL.pop("kind", "stretch")
                _delay = WELLNESS_CTRL.pop("delay", 0)
            if _cmd == "done":
                good_start = nongood_start = sit_start = None
                posture_session["breaks"].append({"t": round(t, 2), "kind": _kind})
                wellness_next_at = now
            elif _cmd == "later":
                wellness_next_at = now + _delay

            if not is_calibrating:
                if detected and current_cls >= 0:
                    away_start = None
                    if sit_start is None:
                        sit_start = now
                    if current_cls == 0:
                        nongood_start = None
                        if good_start is None:
                            good_start = now
                    else:
                        if nongood_start is None:
                            nongood_start = now
                        if now - nongood_start >= GOOD_STREAK_GRACE_S:
                            good_start = None
                        if metrics:
                            wm = worst_metric(metrics)
                            worst_counts[wm] = worst_counts.get(wm, 0) + 1
                elif not detected:
                    if away_start is None:
                        away_start = now
                    if now - away_start >= AWAY_RESET_S:
                        good_start = nongood_start = sit_start = None
                good_dur = (now - good_start) if good_start else 0.0
                sit_dur  = (now - sit_start)  if sit_start  else 0.0
                wellness_due = good_dur >= WELLNESS_GOOD_SECS and now >= wellness_next_at
            else:
                good_dur = sit_dur = 0.0
                wellness_due = False
            focus_metric = max(worst_counts, key=worst_counts.get) if worst_counts else None

            # Mirror the video ONCE here (selfie view). Everything drawn after this
            # line (labels, timer, banners) stays readable, so the browser must NOT
            # mirror the image again with CSS.
            frame = cv2.flip(frame, 1)

            if not detected and not is_calibrating:
                cv2.putText(frame, "Return to camera view",
                            (w // 2 - 170, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255), 2)

            if not is_calibrating:
                cv2.rectangle(frame, (0, 0), (w, 118), (15, 15, 20), -1)
                cv2.putText(frame, label_text, (14, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.15, label_color, 2, cv2.LINE_AA)
                if current_cls >= 0 and len(frame_buffer) == SEQ_LEN:
                    bmax = w - 28
                    cv2.rectangle(frame, (14, 62), (14 + bmax, 71), (45, 45, 45), -1)
                    cv2.rectangle(frame, (14, 62),
                                  (14 + int(confidence * bmax), 71), label_color, -1)
                    cv2.putText(frame, f"Conf {confidence*100:.0f}%",
                                (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (190, 190, 190), 1)
                else:
                    cv2.putText(frame, f"Buffer {len(frame_buffer)}/{SEQ_LEN}",
                                (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (180, 180, 60), 1)
                mm, ss = divmod(int(t), 60)
                snooze_txt = "  SNOOZED" if snoozed else ""
                cv2.putText(frame, f"{mm:02d}:{ss:02d}{snooze_txt}",
                            (w - 185, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                            (200, 130, 30) if snoozed else (155, 155, 155), 1)
                if bad_start:
                    cv2.putText(frame, f"Bad {bad_dur:.0f}s/{BAD_THRESHOLD}s",
                                (w - 175, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                (60, 60, 230), 1, cv2.LINE_AA)

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
                    cv2.rectangle(frame, (0, banner_bottom),
                                  (w, banner_bottom + 44), (20, 70, 90), -1)
                    cv2.putText(frame, f"  PREDICT: {of_warning}",
                                (10, banner_bottom + 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 230, 255), 1)
                if fatigue_msg:
                    cv2.putText(frame, fatigue_msg, (10, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 100), 1)
                if metrics and detected:
                    py = h - 145
                    cv2.rectangle(frame, (0, py - 8), (250, h), (14, 14, 20), -1)
                    cv2.putText(frame, "METRICS", (12, py + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 90, 100), 1)
                    rows = [("Head fwd", "head_forward", ""), ("Shoulder", "shoulder_asym", ""),
                            ("Spinal", "spinal_offset", ""), ("Neck", "neck_angle_deg", "°"),
                            ("Torso", "torso_lean_deg", "°")]
                    for i, (lbl, key, unit) in enumerate(rows):
                        val = metrics[key]
                        c = ((50, 220, 130) if abs(val) < THRESH[key]["moderate"]
                             else (30, 165, 255) if abs(val) < THRESH[key]["bad"]
                             else (60, 60, 230))
                        cv2.putText(frame, f"{lbl:<10} {val:+.3f}{unit}",
                                    (12, py + 22 + i * 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, c, 1, cv2.LINE_AA)

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with lock:
                    LATEST_JPEG = jpeg.tobytes()
                    STATE.update({
                        "detected": detected, "buffer": len(frame_buffer),
                        "cls": -1 if is_calibrating else current_cls,
                        "cls_name": "CALIBRATING" if is_calibrating else label_text,
                        "conf": round(float(confidence), 3), "metrics": metrics,
                        "bad_dur": round(bad_dur, 1), "mod_dur": round(mod_dur, 1),
                        "t": round(t, 1), "bad_alert": bad_alert,
                        "mod_alert": mod_alert, "of_warning": of_warning,
                        "fatigue": fatigue_msg, "snoozed": snoozed,
                        "snooze_left_s": max(0, int(SNOOZE_UNTIL[0] - now)),
                        "flow_mean": round(flow_info["flow_mean"], 5),
                        "bad_rate_pct": round(flow_info["bad_rate_pct"], 1),
                        "alerts": [] if is_calibrating else posture_session["alerts"][-12:],
                        "calibrating": is_calibrating,
                        "good_dur": round(good_dur, 1), "sit_dur": round(sit_dur, 1),
                        "wellness_due": wellness_due,
                        "wellness_after_s": WELLNESS_GOOD_SECS,
                        "breaks_taken": len(posture_session["breaks"]),
                        "focus_metric": focus_metric,
                    })

        except Exception:
            import traceback; traceback.print_exc()
            time.sleep(0.05)   # keep the camera alive; skip just this frame
            continue

    cap.release()
    with lock:
        STATE["running"] = False
        STATE["cls_name"] = "Session ended"
        STATE["wellness_due"] = False
        STATE["good_dur"] = STATE["sit_dur"] = 0.0
    summary = finalise()
    with lock:
        STATE["summary"] = summary
    print("PostureGuard stopped.")

# ── Flask routes ──────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())   # needed for login sessions

# ---- Pages ----
@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard_page"))
    return redirect(url_for("login_page"))

@app.route("/login")
def login_page():
    return render_template("postureguard-login-warm.html")

@app.route("/dashboard")
@login_required
def dashboard_page():
    # Dashboard is analytics/home only. Do NOT start the webcam here.
    # The dashboard's Start Session button calls /api/session/start, which
    # keeps camera ownership with the live monitor and avoids duplicate
    # camera threads / "Could not start the session" races.
    stop_calibration_mode()
    return render_template("postureguard-dashboard-warm.html")

# ---- Session-start compatibility API ---------------------------------
# The monitor page uses /monitor, but the dashboard can start a session
# without navigating through a redesigned dashboard.  These aliases are
# intentionally kept so older/newer dashboard HTML versions can all talk
# to the same backend without breaking the rest of the project.
@app.route("/api/session/start", methods=["POST"])
@app.route("/api/start_session", methods=["POST"])
@app.route("/api/start", methods=["POST"])
@login_required
def api_session_start():
    stop_calibration_mode()
    ensure_camera_running()

    # Give the camera thread a short window to report either success or
    # a real webcam error before replying to the browser.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        with lock:
            running = bool(STATE.get("running"))
            camera_error = STATE.get("camera_error")
        if running:
            return jsonify({"ok": True, "running": True, "redirect": url_for("monitor_page")})
        if camera_error:
            return jsonify({"ok": False, "error": camera_error}), 503
        time.sleep(0.05)

    # The thread may still be warming up. The monitor page will continue
    # polling /api/state, so do not falsely report a hard failure here.
    with lock:
        camera_error = STATE.get("camera_error")
        running = bool(STATE.get("running"))
    if camera_error:
        return jsonify({"ok": False, "error": camera_error}), 503
    return jsonify({"ok": True, "running": running, "redirect": url_for("monitor_page")})

@app.route("/calibrate")
@login_required
def calibrate_page():
    stop_calibration_mode()
    ensure_camera_running()
    return render_template("postureguard-calibrate-warm.html", username=session["user"])

@app.route("/monitor")
@login_required
def monitor_page():
    stop_calibration_mode()
    ensure_camera_running()
    return render_template("postureguard-monitor-warm.html")

@app.route("/logout")
def logout():
    global active_username, baseline, THRESH
    # Completely stop monitoring before showing the login page.
    stop_calibration_mode()
    stop_camera()
    session.clear()
    active_username = None
    baseline = None
    THRESH = build_thresholds(None)
    with lock:
        STATE["thresh"] = THRESH
        STATE["baseline_used"] = False
    return redirect(url_for("login_page"))

# ---- Auth API (one endpoint does both login AND signup) ----
@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    # No format/complexity rules: only require both fields to contain something.
    if not username or not password:
        return jsonify({"ok": False, "error": "Enter a username and password."}), 400

    user = database.get_user_by_username(username)

    global active_username

    if user is None:
        # first time we've seen this username → create the account right now
        database.create_user(username, hash_pw(password))
        session["user"] = username
        active_username = username
        set_active_user(username)
        return jsonify({"ok": True, "redirect": url_for("calibrate_page")})

    if user["password_hash"] != hash_pw(password):
        return jsonify({"ok": False, "error": "Wrong password"}), 401

    # If another account was active in this browser, stop its camera before
    # switching the active user.
    if active_username and active_username != username:
        stop_camera()

    session["user"] = username
    active_username = username
    set_active_user(username)
    next_page = "calibrate_page" if not user["baseline_done"] else "dashboard_page"
    return jsonify({"ok": True, "redirect": url_for(next_page)})

# ---- Baseline calibration API ----
@app.route("/api/calibration/start", methods=["POST"])
@login_required
def api_calibration_start():
    global calibration_mode
    with lock:
        calibration_mode = True
        STATE["calibrating"] = True
        STATE["cls"] = -1
        STATE["cls_name"] = "CALIBRATING"
        STATE["conf"] = 0.0
        STATE["bad_alert"] = False
        STATE["mod_alert"] = False
        STATE["alerts"] = []
    frame_buffer.clear()
    return jsonify({"ok": True})

@app.route("/api/calibration/stop", methods=["POST"])
@login_required
def api_calibration_stop():
    global calibration_mode
    with lock:
        calibration_mode = False
        STATE["calibrating"] = False
        STATE["cls"] = -1
        STATE["cls_name"] = "Warming up…"
    frame_buffer.clear()
    return jsonify({"ok": True})

@app.route("/api/save_baseline", methods=["POST"])
@login_required
def api_save_baseline():
    data = request.get_json(force=True) or {}
    metrics = data.get("metrics")
    if not metrics:
        return jsonify({"ok": False, "error": "No calibration metrics received"}), 400

    username = session["user"]
    database.save_baseline(username, metrics, data.get("frames_used"), data.get("stability_score"))
    database.mark_baseline_done(username)
    set_active_user(username)

    return jsonify({"ok": True, "redirect": url_for("dashboard_page")})

@app.route("/api/skip_baseline", methods=["POST"])
@login_required
def api_skip_baseline():
    username = session["user"]
    database.mark_baseline_done(username)
    set_active_user(username)
    return jsonify({"ok": True, "redirect": url_for("dashboard_page")})

# ---- Session history API (feeds the dashboard) ----
@app.route("/api/history")
@login_required
def api_history():
    # IMPORTANT: read only this logged-in user's sessions from SQLite.
    stored_sessions = database.get_sessions(session["user"])
    sessions_out = []

    for i, s in enumerate(stored_sessions):
        summ = s.get("summary") or {}
        frames = s.get("frames", [])
        t_min, head_forward, torso_lean, alerts = [], [], [], []

        for fr in frames:
            t_min.append(round(fr.get("t", 0) / 60, 3))
            head_forward.append(fr.get("head_forward", 0))
            torso_lean.append(fr.get("torso_lean_deg", 0))

        for a in s.get("alerts", []):
            alerts.append(round(a.get("t", 0) / 60, 3))

        good = float(summ.get("good_pct", s.get("good_pct", 0)) or 0)
        mod = float(summ.get("moderate_pct", s.get("moderate_pct", 0)) or 0)
        bad = float(summ.get("bad_pct", s.get("bad_pct", 0)) or 0)
        duration_s = float(summ.get("duration_s", s.get("duration_s", 0)) or 0)
        score = round(max(0, good + mod * 0.5))

        sessions_out.append({
            "id": s.get("id", i),
            "session_id": s.get("session_id"),
            "date": s.get("start_time", "")[:10],
            "duration_min": round(duration_s / 60, 1),
            "good_pct": good, "moderate_pct": mod, "bad_pct": bad,
            "score": score,
            "baseline_drift_deg": 0,
            "timeline": {
                "t_min": t_min, "head_forward": head_forward,
                "torso_lean": torso_lean, "alerts": alerts
            },
        })

    return jsonify({
        "sessions": sessions_out,
        "baseline": {"head_forward_deg": 0, "torso_lean_deg": 0},
        "streak_days": len(sessions_out),
        "events_by_hour": [],
    })

@app.route("/video_feed")
@login_required
def video_feed():
    def gen():
        while not stop_event.is_set():
            with lock:
                frame = LATEST_JPEG
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/state")
@login_required
def api_state():
    with lock:
        s = dict(STATE)
    return jsonify(s)

@app.route("/api/snooze", methods=["POST"])
@login_required
def api_snooze():
    SNOOZE_UNTIL[0] = time.time() + 300   # replaces the 'S' key
    return jsonify({"ok": True})

@app.route("/api/wellness/ack", methods=["POST"])
@login_required
def api_wellness_ack():
    """The robot coach tells the server what the user did with a break reminder.
       {action:'done', kind:'stretch'|'walk'|'chair'}  -> streak resets, break is logged
       {action:'later', minutes:5}                      -> remind again later"""
    data   = request.get_json(silent=True) or {}
    action = data.get("action", "done")
    with lock:
        if action == "later":
            WELLNESS_CTRL["cmd"]   = "later"
            WELLNESS_CTRL["delay"] = max(1, min(60, int(data.get("minutes", 5)))) * 60
        else:
            WELLNESS_CTRL["cmd"]  = "done"
            WELLNESS_CTRL["kind"] = str(data.get("kind", "stretch"))[:16]
    return jsonify({"ok": True})

@app.route("/api/stop", methods=["POST"])
@app.route("/api/session/stop", methods=["POST"])
@login_required
def api_stop():
    stop_event.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        with lock:
            if STATE.get("summary") is not None:
                return jsonify({"ok": True, "summary": STATE["summary"],
                                "session_file": posture_session_file})
        time.sleep(0.1)
    return jsonify({"ok": True, "summary": None})

if __name__ == "__main__":
    # IMPORTANT: do not start the webcam on the login page.
    # The camera starts only after the user opens Dashboard/Calibrate/Monitor.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)