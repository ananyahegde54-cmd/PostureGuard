"""
PostureGuard — leave-one-subject-out cross-validation

For each subject in posture_data_v2.csv, trains a fresh model on everyone
ELSE and tests only on that subject (who contributes zero frames to
training). Repeats once per subject, then reports per-subject accuracy plus
a pooled accuracy across all folds combined.

This is a genuinely different, harder question than the block-wise held-out
test in train_simple.py/evaluate_simple.py: those measure "can the model
generalize to a new moment in time from a person it already partly trained
on"; this measures "can it generalize to a person it has NEVER seen at all."
Expect a meaningfully lower, more defensible number — that's the point, not
a flaw.

NOTE: this script does not produce your final deployed model. Its only job
is to give you an honest generalization number for your paper/report. Once
you're happy with that number, keep training your actual deployed checkpoint
on ALL subjects combined (train_simple.py / train_lstm.py) — you don't
deploy a model that's missing a sixth of its available data on purpose.

Handles the same Git LFS pointer corruption in posture_data_v2.csv noted
earlier (first 3 lines are LFS metadata, no real header row) and the
"SO11" -> "S011" typo.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

V2_CSV = "posture_data_v2.csv"
SEQUENCE_LENGTH = 30
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 20          # fewer than train_simple.py's 30 since this runs 6x
LR = 1e-3
CLASS_NAMES = ["good", "moderate", "bad"]

LANDMARK_COLS = []
for i in range(33):
    LANDMARK_COLS += [f"x{i}", f"y{i}", f"z{i}"]
V2_COLUMNS = (
    ["timestamp", "subject_id", "session_id", "camera_angle", "angle_code", "age_bracket",
     "posture_class"] + LANDMARK_COLS + ["frame_quality", "label"]
)
SUBJECT_ID_FIXES = {"SO11": "S011"}   # known typo — O instead of 0


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


def load_v2():
    df = pd.read_csv(V2_CSV, skiprows=3, header=None, names=V2_COLUMNS)
    df["subject_id"] = df["subject_id"].replace(SUBJECT_ID_FIXES)
    return df


def make_windows_grouped(df, group_col, seq_len=SEQUENCE_LENGTH, stride=1):
    """Window within each group (session_id) only — never across a session boundary."""
    X, y = [], []
    for _, g in df.groupby(group_col, sort=False):
        landmarks = g[LANDMARK_COLS].values.astype(np.float32)
        labels = g["label"].values.astype(np.int64)
        n = len(landmarks)
        for i in range(0, n - seq_len + 1, stride):
            X.append(landmarks[i:i + seq_len])
            y.append(labels[i + seq_len - 1])
    return np.array(X), np.array(y)


def train_one_fold(X_train, y_train, epochs=EPOCHS):
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = PostureLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def evaluate_fold(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        preds = torch.argmax(logits, dim=1).numpy()
    return preds


def main():
    print(f"Loading {V2_CSV}...")
    df = load_v2()
    subjects = sorted(df["subject_id"].unique())
    print(f"Subjects found: {subjects}")

    fold_results = []
    all_preds, all_true = [], []

    for held_out in subjects:
        train_df = df[df["subject_id"] != held_out]
        test_df = df[df["subject_id"] == held_out]

        X_train, y_train = make_windows_grouped(train_df, "session_id", stride=1)
        X_test, y_test = make_windows_grouped(test_df, "session_id", stride=SEQUENCE_LENGTH)

        print(f"\n{'='*60}\nFold: held out {held_out}\n{'='*60}")
        print(f"Train windows: {len(X_train)} (from {len(subjects)-1} other subjects)")
        print(f"Test windows: {len(X_test)} (non-overlapping, {held_out} only, "
              f"never seen in training)")

        if len(X_test) == 0:
            print(f"Skipping {held_out} — not enough frames for even one test window.")
            continue

        model = train_one_fold(X_train, y_train)
        preds = evaluate_fold(model, X_test, y_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy on {held_out}: {acc:.4f}")

        fold_results.append({"subject": held_out, "n_test": len(X_test), "accuracy": acc})
        all_preds.extend(preds.tolist())
        all_true.extend(y_test.tolist())

    print(f"\n{'='*70}\nPer-subject LOSO results\n{'='*70}")
    print(f"{'Subject':<10}{'N test windows':>16}{'Accuracy':>12}")
    for r in fold_results:
        print(f"{r['subject']:<10}{r['n_test']:>16}{r['accuracy']:>12.4f}")

    accs = [r["accuracy"] for r in fold_results]
    print(f"\nMean accuracy across subjects: {np.mean(accs):.4f}  (std: {np.std(accs):.4f})")

    print(f"\n{'='*70}\nPooled results (all held-out predictions combined)\n{'='*70}")
    pooled_acc = accuracy_score(all_true, all_preds)
    print(f"Pooled accuracy: {pooled_acc:.4f}  ({len(all_true)} total windows)")
    print("\nPer-class report:")
    print(classification_report(all_true, all_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(all_true, all_preds))

    print("\nThis is a genuine new-person generalization number: every prediction above "
          "came from a model that had never seen that subject's data during training. "
          "Report BOTH the mean-across-subjects and the pooled number in your paper — "
          "they can legitimately differ when subjects have very different amounts of data.")


if __name__ == "__main__":
    main()
