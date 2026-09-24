# app.py
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import json
import os
import time
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ── Load LSTM model ────────────────────────────────────────────
class PostureLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
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

# ── Load baseline ──────────────────────────────────────────────
BASELINE_FILE = "user_baseline.json"
baseline = None
if os.path.exists(BASELINE_FILE):
    with open(BASELINE_FILE) as f:
        baseline = json.load(f).get("metrics", None)

# ── MediaPipe ──────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.65, min_tracking_confidence=0.65)

SEQ_LEN = 30
frame_buffer = deque(maxlen=SEQ_LEN)

CLS = {
    0: {"name": "GOOD POSTURE", "color": "#32DC82", "short": "good"},
    1: {"name": "MODERATE POSTURE", "color": "#1EA5FF", "short": "moderate"},
    2: {"name": "BAD POSTURE", "color": "#3C3CE6", "short": "bad"},
}

def compute_metrics(lms):
    def pt(i):
        return np.array([lms[i].x, lms[i].y, lms[i].z])
    ear = (pt(7) + pt(8)) / 2
    sh = (pt(11) + pt(12)) / 2
    hip = (pt(23) + pt(24)) / 2
    l_sh, r_sh = pt(11), pt(12)
    hf = float(sh[0] - ear[0])
    sa = float(abs(l_sh[1] - r_sh[1]))
    so = float(abs(sh[0] - hip[0]))
    vn = sh[:2] - ear[:2]
    na = float(np.degrees(np.arctan2(abs(vn[0]), abs(vn[1]) + 1e-6)))
    vt = hip[:2] - sh[:2]
    tl = float(np.degrees(np.arctan2(abs(vt[0]), abs(vt[1]) + 1e-6)))
    return {
        "head_forward": hf,
        "shoulder_asym": sa,
        "spinal_offset": so,
        "neck_angle_deg": na,
        "torso_lean_deg": tl
    }

def get_thresholds():
    if baseline:
        return {
            "head_forward": {"moderate": baseline["head_forward"]["warn_moderate"], 
                             "bad": baseline["head_forward"]["warn_bad"]},
            "shoulder_asym": {"moderate": baseline["shoulder_asym"]["warn_moderate"],
                              "bad": baseline["shoulder_asym"]["warn_bad"]},
            "spinal_offset": {"moderate": baseline["spinal_offset"]["warn_moderate"],
                              "bad": baseline["spinal_offset"]["warn_bad"]},
            "neck_angle_deg": {"moderate": baseline["neck_angle_deg"]["warn_moderate"],
                               "bad": baseline["neck_angle_deg"]["warn_bad"]},
            "torso_lean_deg": {"moderate": baseline["torso_lean_deg"]["warn_moderate"],
                               "bad": baseline["torso_lean_deg"]["warn_bad"]}
        }
    return {
        "head_forward": {"moderate": 0.030, "bad": 0.060},
        "shoulder_asym": {"moderate": 0.030, "bad": 0.060},
        "spinal_offset": {"moderate": 0.040, "bad": 0.080},
        "neck_angle_deg": {"moderate": 15.0, "bad": 25.0},
        "torso_lean_deg": {"moderate": 10.0, "bad": 18.0}
    }

THRESH = get_thresholds()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/status')
def status():
    """Return current posture status as JSON"""
    return jsonify({
        "status": "running",
        "baseline_loaded": baseline is not None,
        "model_loaded": True
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'message': 'Connected to posture detection server'})

@socketio.on('start_detection')
def handle_start_detection():
    """Start posture detection stream"""
    print('Starting posture detection...')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        emit('error', {'message': 'Cannot open webcam'})
        return
    
    frame_buffer.clear()
    t0 = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        now = time.time()
        t = now - t0
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        data = {
            'timestamp': t,
            'person_detected': False,
            'class': -1,
            'class_name': 'No person detected',
            'confidence': 0,
            'metrics': {},
            'alert': None
        }
        
        if results.pose_landmarks:
            data['person_detected'] = True
            kp = [v for lm in results.pose_landmarks.landmark 
                  for v in (lm.x, lm.y, lm.z)]
            frame_buffer.append(kp)
            metrics = compute_metrics(results.pose_landmarks.landmark)
            data['metrics'] = metrics
            
            if len(frame_buffer) == SEQ_LEN:
                tensor = torch.from_numpy(
                    np.array(frame_buffer, dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1).squeeze().tolist()
                cls = int(np.argmax(probs))
                confidence = probs[cls]
                data['class'] = cls
                data['class_name'] = CLS[cls]['name']
                data['confidence'] = confidence
                data['color'] = CLS[cls]['color']
                
                # Check for alerts
                if cls == 2:
                    data['alert'] = {
                        'type': 'bad',
                        'message': 'Fix your posture!',
                        'tip': 'Tuck chin back — ears over shoulders'
                    }
                elif cls == 1:
                    data['alert'] = {
                        'type': 'moderate',
                        'message': 'Posture drifting — sit straight'
                    }
        
        emit('posture_update', data)
        socketio.sleep(0.05)  # 20 fps
        
        if not ret:
            break
    
    cap.release()
    emit('detection_stopped', {'message': 'Detection stopped'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)