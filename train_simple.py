"""
PostureGuard — training script (leak-free version)

Fixes a real methodological bug in the previous version: sliding windows were
built with stride=1 over the WHOLE flat frame sequence, then split into
train/test with a random row-level train_test_split. Two consequences:
  1. Adjacent windows share 29/30 frames, so many near-duplicate windows
     ended up on both sides of the split — the model was tested on samples
     it had essentially already seen, not on genuinely unseen data.
  2. A window could span the BOUNDARY between two different recording
     blocks (e.g. the last few frames of a "good" block and the first few
     of a "moderate" block), producing a window whose middle frames don't
     match its own label.
This version splits BEFORE windowing, at the level of contiguous same-label
recording blocks (each block = one uninterrupted collect_data_simple.py
recording), and windows each block separately so no window ever crosses a
block boundary and no test window overlaps a train window. Test windows are
also non-overlapping (stride = seq_len), so no two test windows share frames
with each other either. The result is a genuinely held-out accuracy number,
not one inflated by leakage.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_FILE = "posture_data_final.csv"
MODEL_OUT = "posture_model_final.pth"
SEQUENCE_LENGTH = 30
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
TEST_FRACTION = 0.2   # last 20% of each contiguous block held out as test


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
    """Contiguous runs of the same label -> list of (start, end) index ranges."""
    blocks = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            blocks.append((start, i))
            start = i
    return blocks


def split_into_segments(landmarks, labels, test_fraction=TEST_FRACTION):
    """Cut each contiguous block at (1 - test_fraction) through its length.
    Returns train_segments, test_segments — each a list of (landmarks, labels)
    arrays, one pair per block-portion. Windowing each segment separately
    (below) guarantees no window crosses a block boundary or the train/test cut."""
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
    """Slide windows WITHIN each segment only — never across segment boundaries."""
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

    train_segments, test_segments, blocks = split_into_segments(landmarks, labels)
    print(f"Found {len(blocks)} contiguous recording blocks "
          f"(avg length {np.mean([e - s for s, e in blocks]):.0f} frames)")
    print(f"Split: {len(train_segments)} train segments, {len(test_segments)} test segments "
          f"(last {TEST_FRACTION*100:.0f}% of each block held out)")

    # train windows: stride=1 (overlap is fine WITHIN train — it's never compared
    # against test, so it can't leak); test windows: stride=seq_len (non-overlapping,
    # so no two test windows share a single frame with each other either).
    X_train, y_train = make_windows_from_segments(train_segments, stride=1)
    X_test, y_test = make_windows_from_segments(test_segments, stride=SEQUENCE_LENGTH)
    print(f"Train windows: {len(X_train)} (overlapping, stride=1)")
    print(f"Test windows: {len(X_test)} (non-overlapping, stride={SEQUENCE_LENGTH}) — "
          f"genuinely held out, zero frame overlap with train or across test windows")

    if len(X_test) == 0:
        print("WARNING: no test windows produced — your blocks may be shorter than "
              f"{SEQUENCE_LENGTH} frames after the {TEST_FRACTION*100:.0f}% cut. "
              "Record longer sessions per class, or lower SEQUENCE_LENGTH.")
        return

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = PostureLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nTraining...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{EPOCHS} — loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nSaved {MODEL_OUT}")

    print("\n=== Held-out test evaluation (leak-free) ===")
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = torch.argmax(model(xb), dim=1)
            all_preds.extend(preds.tolist())
            all_true.extend(yb.tolist())

    acc = accuracy_score(all_true, all_preds)
    print(f"Test accuracy: {acc:.4f}")
    print("\nPer-class report (0=good, 1=moderate, 2=bad):")
    print(classification_report(all_true, all_preds, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(all_true, all_preds))
    print("\nThis number reflects a genuine held-out split: test windows never share a "
          "frame with any train window, never cross a recording-block boundary, and "
          "don't overlap each other either.")


if __name__ == "__main__":
    main()
