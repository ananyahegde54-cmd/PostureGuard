"""
PostureGuard — model evaluation script

Measures more than just overall accuracy: per-class precision/recall/F1, a confusion
matrix, accuracy sliced by camera_angle and age_bracket (posture_data_v2.csv only —
the legacy posture_data.csv has no metadata columns), accuracy on the frozen canary
set alone (regression check), and per-window inference latency.

============================================================================
STEP 1 — PLUG IN YOUR MODEL(S) HERE. Nothing else below needs editing unless
your CSV column names or file paths differ from what's listed in CONFIG.
============================================================================
"""

import numpy as np
import pandas as pd
import torch
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# net/st_gcn.py and graph/mediapipe_graph.py live under stgcn/, but this script runs from
# the repo root — add stgcn/ to sys.path so "from net.st_gcn import Model" resolves.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stgcn"))

# TODO: import your actual model class(es). The 3-class PostureLSTM below is copied
# from realtime_posture.py, which is the architecture that actually matches your
# saved posture_model.pth (train_lstm.py in the repo defines a DIFFERENT, binary
# version — it's stale and will break on 3-class data; fix it before you retrain).
import torch.nn as nn

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


def load_models():
    """
    Return a dict of {model_name: (model, predict_fn)}.
    predict_fn(model, window) -> int class prediction, given a window of shape
    (SEQUENCE_LENGTH, NUM_FEATURES).
    """
    lstm_model = PostureLSTM()
    lstm_model.load_state_dict(torch.load("posture_model.pth", map_location="cpu"))
    lstm_model.eval()

    def lstm_predict(model, window):
        with torch.no_grad():
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
            probs = torch.softmax(model(x), dim=1)
            return int(torch.argmax(probs, dim=1).item())

    models = {"LSTM": (lstm_model, lstm_predict)}

    # ---------------- STGCN ----------------
    if EVALUATE_STGCN:
        # Real class is net.st_gcn.Model (net/st_gcn.py in the postguard-stgcn branch).
        # models/stgcn.py in that same branch is an incomplete duplicate — cut off
        # mid-definition, no forward() — and isn't used anywhere; don't import it.
        from net.st_gcn import Model as STGCNModel

        stgcn_model = STGCNModel(
            in_channels=3,
            num_class=3,
            graph_args={"layout": "mediapipe", "strategy": "uniform"},
            edge_importance_weighting=True,
        )
        stgcn_model.load_state_dict(torch.load("stgcn/models/stgcn_best.pth", map_location="cpu"))
        stgcn_model.eval()

        def stgcn_predict(model, window):
            # window: (120, 33, 3) -> (N=1, C=3, T=120, V=33, M=1), exactly as in live_predict.py
            with torch.no_grad():
                x = np.transpose(window, (2, 0, 1))       # (3, 120, 33)
                x = np.expand_dims(x, axis=-1)             # (3, 120, 33, 1)
                x = np.expand_dims(x, axis=0)               # (1, 3, 120, 33, 1)
                x = torch.tensor(x, dtype=torch.float32)
                probs = torch.softmax(model(x), dim=1)
                return int(torch.argmax(probs, dim=1).item())

        models["STGCN"] = (stgcn_model, stgcn_predict)

    return models


def load_stgcn_npy_dataset():
    """
    STGCN's own dataset: one (120, 33, 3) array per .npy file under
    STGCN_DATA_DIR/<class_name>/. No subject_id/camera_angle/age_bracket exists for
    these samples, so this returns X, y only (no subgroup breakdown possible).
    """
    X, y = [], []
    for class_name, label in STGCN_CLASSES.items():
        folder = os.path.join(STGCN_DATA_DIR, class_name)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".npy"):
                X.append(np.load(os.path.join(folder, fname)))
                y.append(label)
    return np.array(X), np.array(y)


# ============================================================================
# CONFIG
# ============================================================================

EVALUATE_STGCN = False        # flip to True once stgcn/ is set up next to this script

SEQUENCE_LENGTH = 30          # LSTM window length. STGCN uses a different length (120) —
                               # see MODEL_SEQUENCE_LENGTHS below, used per-model in evaluate().
MODEL_SEQUENCE_LENGTHS = {"LSTM": 30, "STGCN": 120}
NUM_LANDMARKS = 33
COORDS_PER_LANDMARK = 3       # x, y, z
NUM_FEATURES = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 99

V2_CSV = "posture_data_v2.csv"          # new data, has metadata columns
LEGACY_CSV = "posture_data.csv"         # frozen canary/baseline, landmarks + label only

# STGCN's dataset is NOT in the CSVs above — it's a separate pipeline: one .npy file per
# sample (shape (120, 33, 3)) under STGCN_DATA_DIR/<class_name>/, with no subject_id,
# camera_angle, or age_bracket recorded. Evaluated separately from the LSTM's CSV-based
# test set below until that pipeline gets the same metadata tracking as collect_data.py.
STGCN_DATA_DIR = "stgcn/data/new_data"
STGCN_CLASSES = {"good": 0, "moderate": 1, "bad": 2}

TEST_SUBJECT_IDS = []  # TODO: fill with the subject_ids held out for your test split,
                        # e.g. ["S004", "S009", "S013"]. Leave empty to evaluate on
                        # the whole v2 file (only do this if you've already split
                        # v2 into a separate test-only CSV beforehand).


# ============================================================================
# Data loading — handles both CSV schemas
# ============================================================================

def load_v2_csv(path):
    df = pd.read_csv(path)
    landmark_cols = []
    for i in range(NUM_LANDMARKS):
        landmark_cols += [f"x{i}", f"y{i}", f"z{i}"]
    return df, landmark_cols


def load_legacy_csv(path):
    df = pd.read_csv(path)
    landmark_cols = [c for c in df.columns if c != "label"]
    extra = pd.DataFrame({
        "subject_id": ["legacy"] * len(df),
        "camera_angle": [None] * len(df),
        "age_bracket": [None] * len(df),
    })
    df = pd.concat([df, extra], axis=1)
    return df, landmark_cols


def make_windows(df, landmark_cols, group_cols, seq_len=SEQUENCE_LENGTH):
    """
    Stride-1 sliding windows within each group (a group = one contiguous recording —
    session_id if present, else a contiguous same-label run for legacy data), matching
    how train_lstm.py builds sequences (make_sequences) and how realtime_posture.py's
    deque buffer slides. Window label = label of the LAST frame in the window, same
    convention as make_sequences() in train_lstm.py.

    Note: stride-1 windows overlap heavily (window i and i+1 share 29/30 frames), so
    per-window accuracy numbers will look smoother/higher than per-event accuracy would.
    That's consistent with your existing training script, but worth remembering when you
    move to subject-wise splitting: overlapping windows from the SAME recording must never
    be split across train and test, only across different sessions/subjects, or you'll
    leak near-duplicate windows across the split.
    """
    X, y, meta = [], [], []
    for _, group in df.groupby(group_cols, sort=False):
        landmarks = group[landmark_cols].values
        labels = group["label"].values
        for i in range(len(group) - seq_len):
            window = landmarks[i:i + seq_len]
            window_label = int(labels[i + seq_len - 1])
            X.append(window)
            y.append(window_label)
            meta.append({
                "subject_id": group["subject_id"].iloc[0],
                "camera_angle": group["camera_angle"].iloc[0],
                "age_bracket": group["age_bracket"].iloc[0],
            })
    return np.array(X), np.array(y), pd.DataFrame(meta)


def build_legacy_groups(df):
    # legacy CSV has no session_id — approximate session boundaries as contiguous
    # runs of the same label (this matches how the original collect_data.py recorded
    # data: one uninterrupted block per class).
    df = df.copy()
    df["_group"] = (df["label"] != df["label"].shift()).cumsum()
    return df


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model_name, model, predict_fn, X, y, meta=None, label_desc=""):
    preds = []
    latencies_ms = []
    for window in X:
        t0 = time.perf_counter()
        pred = predict_fn(model, window)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        preds.append(pred)
    preds = np.array(preds)

    print(f"\n{'='*70}\n{model_name} — {label_desc}\n{'='*70}")
    print(f"Windows evaluated: {len(y)}")
    print(f"Overall accuracy: {accuracy_score(y, preds):.4f}")
    print(f"Avg inference latency: {np.mean(latencies_ms):.2f} ms/window "
          f"(p95: {np.percentile(latencies_ms, 95):.2f} ms)")
    print("\nPer-class report:")
    print(classification_report(y, preds, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y, preds))

    if meta is not None and meta["camera_angle"].notna().any():
        print("\nAccuracy by camera_angle:")
        meta_eval = meta.copy()
        meta_eval["correct"] = (preds == y)
        print(meta_eval.groupby("camera_angle")["correct"].mean().to_string())

        print("\nAccuracy by age_bracket:")
        print(meta_eval.groupby("age_bracket")["correct"].mean().to_string())

        print("\nAccuracy by camera_angle x age_bracket (watch for weak cells):")
        print(meta_eval.groupby(["camera_angle", "age_bracket"])["correct"].mean().to_string())

    return preds


def main():
    models = load_models()

    # --- Held-out test split from the new, angle/age-tagged CSV data (LSTM path) ---
    if TEST_SUBJECT_IDS:
        df_v2, landmark_cols_v2 = load_v2_csv(V2_CSV)
        test_df = df_v2[df_v2["subject_id"].isin(TEST_SUBJECT_IDS)]
        for name, (model, predict_fn) in models.items():
            if name == "STGCN":
                continue  # STGCN doesn't consume this CSV format — see below
            seq_len = MODEL_SEQUENCE_LENGTHS.get(name, SEQUENCE_LENGTH)
            X_test, y_test, meta_test = make_windows(
                test_df, landmark_cols_v2, group_cols=["session_id"], seq_len=seq_len
            )
            evaluate(name, model, predict_fn, X_test, y_test, meta_test,
                      label_desc="held-out test set (new data)")
    else:
        print("TEST_SUBJECT_IDS is empty — skipping the v2 held-out test evaluation. "
              "Fill it in with the subject_ids from your test split.")

    # --- Canary/legacy set — regression check against the original baseline (LSTM path) ---
    df_legacy, landmark_cols_legacy = load_legacy_csv(LEGACY_CSV)
    df_legacy = build_legacy_groups(df_legacy)
    for name, (model, predict_fn) in models.items():
        if name == "STGCN":
            continue
        seq_len = MODEL_SEQUENCE_LENGTHS.get(name, SEQUENCE_LENGTH)
        X_canary, y_canary, meta_canary = make_windows(
            df_legacy, landmark_cols_legacy, group_cols=["_group"], seq_len=seq_len
        )
        evaluate(name, model, predict_fn, X_canary, y_canary, meta_canary,
                  label_desc="canary/legacy set (regression check)")

    # --- STGCN's own dataset (separate pipeline, no CSV/metadata) ---
    if "STGCN" in models:
        X_stgcn, y_stgcn = load_stgcn_npy_dataset()
        if len(X_stgcn) == 0:
            print(f"\nNo .npy files found under {STGCN_DATA_DIR} — skipping STGCN evaluation.")
        else:
            model, predict_fn = models["STGCN"]
            evaluate("STGCN", model, predict_fn, X_stgcn, y_stgcn, meta=None,
                      label_desc=f"native .npy dataset ({len(X_stgcn)} samples, "
                                 f"no subject/angle/age metadata available)")
            print("\nNOTE: this STGCN number is NOT directly comparable to the LSTM numbers "
                  "above — different dataset, no held-out-by-subject split enforced here "
                  "(check whether train_stgcn.py's split leaks subjects across train/test), "
                  "and no angle/age subgroup breakdown is possible until collect_dataset_v2.py "
                  "records that metadata the way collect_data.py now does.")


if __name__ == "__main__":
    main()