"""
PostureGuard — evaluation script (leak-free version)

Evaluates the already-trained posture_model_final.pth against a held-out test
set built the SAME way train_simple.py builds it: split by contiguous
recording block (not by random window), non-overlapping test windows, no
window crossing a block boundary. This reproduces train_simple.py's exact
test set deterministically — no random seed needed, since the split is
defined entirely by block structure, not randomness.
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_FILE = "posture_data_final.csv"
MODEL_FILE = "posture_model_final.pth"
SEQUENCE_LENGTH = 30
NUM_CLASSES = 3
CLASS_NAMES = ["good", "moderate", "bad"]
TEST_FRACTION = 0.2   # must match train_simple.py


class PostureLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=2,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.norm(out[:, -1, :])))


def find_blocks(labels):
    blocks = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            blocks.append((start, i))
            start = i
    return blocks


def split_into_segments(landmarks, labels, test_fraction=TEST_FRACTION):
    blocks = find_blocks(labels)
    train_segments, test_segments = [], []
    for (s, e) in blocks:
        n = e - s
        cut = s + int(n * (1 - test_fraction))
        if cut > s:
            train_segments.append((landmarks[s:cut], labels[s:cut]))
        if e > cut:
            test_segments.append((landmarks[cut:e], labels[cut:e]))
    return train_segments, test_segments, blocks


def make_windows_from_segments(segments, seq_len=SEQUENCE_LENGTH, stride=1):
    X, y = [], []
    for seg_landmarks, seg_labels in segments:
        n = len(seg_landmarks)
        for i in range(0, n - seq_len + 1, stride):
            X.append(seg_landmarks[i:i + seq_len])
            y.append(seg_labels[i + seq_len - 1])
    return np.array(X), np.array(y)


def main():
    print(f"Loading {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    landmark_cols = [c for c in df.columns if c != "label"]
    landmarks = df[landmark_cols].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)
    print(f"Rows: {len(df)}, label distribution: {pd.Series(labels).value_counts().to_dict()}")

    _, test_segments, blocks = split_into_segments(landmarks, labels)
    print(f"Found {len(blocks)} contiguous recording blocks")

    X_test, y_test = make_windows_from_segments(test_segments, stride=SEQUENCE_LENGTH)
    print(f"Test windows: {len(X_test)} (non-overlapping, block-safe, "
          f"last {TEST_FRACTION*100:.0f}% of each block)")

    if len(X_test) == 0:
        print("No test windows produced — recording blocks are too short. "
              "See train_simple.py's warning for the same issue.")
        return

    print(f"\nLoading model from {MODEL_FILE}...")
    model = PostureLSTM()
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()

    preds = []
    latencies_ms = []
    with torch.no_grad():
        for window in X_test:
            t0 = time.perf_counter()
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            probs = torch.softmax(model(x), dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            preds.append(pred)
    preds = np.array(preds)

    acc = accuracy_score(y_test, preds)
    print(f"\n{'='*60}")
    print(f"Test accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Avg inference latency: {np.mean(latencies_ms):.2f} ms/window "
          f"(p95: {np.percentile(latencies_ms, 95):.2f} ms)")
    print(f"{'='*60}")
    print("\nPer-class report:")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted, order=good/moderate/bad):")
    print(confusion_matrix(y_test, preds))

    print("\nThis test set has zero frame overlap with training data, never crosses a "
          "recording-block boundary, and its windows don't overlap each other either — "
          "a genuine held-out evaluation.")


if __name__ == "__main__":
    main()
