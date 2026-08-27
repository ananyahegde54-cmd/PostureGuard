# app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import json
import base64
import os
import time
from datetime import datetime
import hashlib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
socketio = SocketIO(app, cors_allowed_origins="*")
active_streams = {}

# ── User Management ────────────────────────────────────────────
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ── Baseline Management ───────────────────────────────────────
BASELINE_FILE = "user_baseline.json"

def load_baseline():
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, 'r') as f:
            return json.load(f)
    return None

def save_baseline(baseline_data):
    with open(BASELINE_FILE, 'w') as f:
        json.dump(baseline_data, f, indent=2)

# ── Load LSTM model ────────────────────────────────────────────
class PostureLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.norm(out[:, -1, :])))

model = None
try:
    model = PostureLSTM()
    model.load_state_dict(torch.load("posture_model.pth", map_location="cpu"))
    model.eval()
    print("✅ LSTM model loaded.")
except Exception as e:
    print(f"Model not loaded: {e}")

# ── MediaPipe ──────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.65, min_tracking_confidence=0.65)

SEQ_LEN = 30
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

def encode_frame(frame):
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not success:
        return None
    return base64.b64encode(buffer).decode('ascii')

# ── Routes ─────────────────────────────────────────────────────

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            return render_template('login.html', error='Please fill in all fields')
        
        users = load_users()
        
        if username in users:
            if users[username]['password'] == hash_password(password):
                session['user'] = username
                session['baseline_done'] = users[username].get('baseline_done', False)
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error='Invalid password')
        else:
            # Create new user
            users[username] = {
                'password': hash_password(password),
                'baseline_done': False,
                'created_at': datetime.now().isoformat()
            }
            save_users(users)
            session['user'] = username
            session['baseline_done'] = False
            return redirect(url_for('calibrate'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    baseline = load_baseline()
    return render_template('dashboard.html', 
                         username=session['user'],
                         baseline_exists=baseline is not None)

@app.route('/calibrate')
def calibrate():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('calibrate.html', username=session['user'])

@app.route('/api/save_baseline', methods=['POST'])
def save_baseline_route():
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.json
    if not data or 'metrics' not in data:
        return jsonify({'error': 'Invalid data'}), 400
    
    # Update user record
    users = load_users()
    if session['user'] in users:
        users[session['user']]['baseline_done'] = True
        save_users(users)
        session['baseline_done'] = True
    
    # Save baseline
    baseline_data = {
        "calibrated_at": datetime.now().isoformat(),
        "username": session['user'],
        "frames_used": data.get('frames_used', 0),
        "stability_score": data.get('stability_score', 0),
        "metrics": data['metrics'],
        "notes": "Personal baseline for posture detection"
    }
    save_baseline(baseline_data)
    
    return jsonify({'success': True})

@app.route('/api/check_baseline')
def check_baseline():
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    baseline = load_baseline()
    users = load_users()
    done = users.get(session['user'], {}).get('baseline_done', False)
    
    return jsonify({
        'baseline_exists': baseline is not None,
        'baseline_done': done
    })

# ── Socket.IO Events ──────────────────────────────────────────

@socketio.on('connect')
def handle_connect():
    print(f'✅ Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    active_streams.pop(request.sid, None)

@socketio.on('stop_detection')
def handle_stop_detection():
    active_streams[request.sid] = False

@socketio.on('start_calibration')
def handle_start_calibration():
    """Stream live metrics to the calibration page until it asks us to stop."""
    print(f'📐 Starting calibration for {request.sid}')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        emit('error', {'message': 'Cannot open webcam'})
        return

    active_streams[request.sid] = True
    try:
        while active_streams.get(request.sid, False):
            ret, frame = cap.read()
            if not ret:
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                emit('calibration_update', {
                    'metrics': compute_metrics(results.pose_landmarks.landmark),
                    'frame': encode_frame(frame)
                })
            else:
                encoded_frame = encode_frame(frame)
                if encoded_frame:
                    emit('calibration_update', {'metrics': {}, 'frame': encoded_frame.hex()})
            socketio.sleep(0.05)
    finally:
        cap.release()
        active_streams.pop(request.sid, None)
        emit('calibration_stopped', {'message': 'Calibration stopped'})

@socketio.on('stop_calibration')
def handle_stop_calibration():
    active_streams[request.sid] = False

@socketio.on('start_detection')
def handle_start_detection():
    """Start posture detection stream"""
    print('📷 Starting posture detection...')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        emit('error', {'message': 'Cannot open webcam'})
        return

    frame_buffer = deque(maxlen=SEQ_LEN)
    active_streams[request.sid] = True
    t0 = time.time()
    baseline = load_baseline()
    
    # Get thresholds from baseline if available
    thresholds = None
    if baseline and 'metrics' in baseline:
        thresholds = {}
        for key, values in baseline['metrics'].items():
            thresholds[key] = {
                'moderate': values.get('warn_moderate', 0.03),
                'bad': values.get('warn_bad', 0.06)
            }
    
    try:
        while active_streams.get(request.sid, False):
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
                'alert': None,
                'thresholds': thresholds
            }
            encoded_frame = encode_frame(frame)
            if encoded_frame:
                    data['frame'] = encoded_frame

            if results.pose_landmarks:
                data['person_detected'] = True
                kp = [v for lm in results.pose_landmarks.landmark
                      for v in (lm.x, lm.y, lm.z)]
                frame_buffer.append(kp)
                metrics = compute_metrics(results.pose_landmarks.landmark)
                data['metrics'] = metrics

                if len(frame_buffer) == SEQ_LEN and model is not None:
                    tensor = torch.from_numpy(
                        np.array(frame_buffer, dtype=np.float32)).unsqueeze(0)
                    with torch.no_grad():
                        probs = torch.softmax(model(tensor), dim=1).squeeze().tolist()
                    cls = int(np.argmax(probs))
                    confidence = probs[cls]
                    data['class'] = cls
                    data['class_name'] = CLS[cls]['name']
                    data['confidence'] = confidence

                    # Check for alerts using thresholds
                    if thresholds and cls == 2:
                        data['alert'] = {
                            'type': 'bad',
                            'message': 'Bad posture detected!',
                            'tip': 'Sit up straight - tuck chin back'
                        }
                    elif thresholds and cls == 1:
                        data['alert'] = {
                            'type': 'moderate',
                            'message': 'Posture drifting',
                            'tip': 'Straighten your back'
                        }

            emit('posture_update', data)
            socketio.sleep(0.05)
    finally:
        cap.release()
        active_streams.pop(request.sid, None)
        emit('detection_stopped', {'message': 'Detection stopped'})

if __name__ == '__main__':
    # Create default users if none exist
    if not os.path.exists(USERS_FILE):
        save_users({})
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)