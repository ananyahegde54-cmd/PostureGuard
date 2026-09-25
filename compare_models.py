"""
PostureGuard — LSTM vs STGCN comparison for the research paper

Evaluates both models on their OWN held-out test data (they don't currently
share a test set — see the limitation note this script prints), reports the
same metrics for each, runs a bootstrap comparison to check whether any
accuracy gap is statistically meaningful, and writes a paper-ready table.

Honesty check this script enforces: it will NOT let you present these numbers
as directly comparable without printing the sample-size/data-source caveat
every time. Put that caveat in your paper's methodology section, not just in
this script's output.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

# net/st_gcn.py lives under stgcn/ — add it to sys.path (only needed if RUN_STGCN=True)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stgcn"))

RUN_STGCN = True  # set False to only produce the LSTM half of the table

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LSTM_CSV = "posture_data_combined.csv"       # x0..z32,label format
LSTM_MODEL_PATH = "posture_model_final.pth"  # or posture_model.pth, whichever you trained
LSTM_SEQ_LEN = 30

STGCN_DATA_DIR = "data/stgcn_dataset"
STGCN_MODEL_PATH = "models/stgcn_best.pth"
STGCN_SEQ_LEN = 120
STGCN_CLASSES = {"good": 0, "moderate": 1, "bad": 2}

CLASS_NAMES = ["good", "moderate", "bad"]
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class PostureLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=2,
                 num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.norm(out[:, -1, :])))


def make_sequences(landmarks, labels, seq_len):
    X, y = [], []
    for i in range(len(landmarks) - seq_len):
        X.append(landmarks[i:i + seq_len])
        y.append(labels[i + seq_len - 1])
    return np.array(X), np.array(y)


def load_stgcn_npy_dataset():
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_inference(predict_fn, X):
    preds, latencies_ms = [], []
    for window in X:
        t0 = time.perf_counter()
        preds.append(predict_fn(window))
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    return np.array(preds), np.array(latencies_ms)


def bootstrap_accuracy_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def summarize(model_name, y_true, y_pred, latencies_ms):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    lo, hi = bootstrap_accuracy_ci(np.array(y_true), np.array(y_pred))
    result = {
        "model": model_name,
        "n_test_samples": len(y_true),
        "accuracy": acc,
        "accuracy_95ci": (lo, hi),
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "avg_latency_ms": np.mean(latencies_ms),
    }

    print(f"\n{'='*70}\n{model_name}\n{'='*70}")
    print(f"Test samples: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}  (95% bootstrap CI: [{lo:.4f}, {hi:.4f}])")
    print(f"Macro precision/recall/F1: {prec:.4f} / {rec:.4f} / {f1:.4f}")
    print(f"Avg inference latency: {np.mean(latencies_ms):.2f} ms/window")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))
    return result


def significance_note(result_a, result_b):
    lo_a, hi_a = result_a["accuracy_95ci"]
    lo_b, hi_b = result_b["accuracy_95ci"]
    overlap = not (hi_a < lo_b or hi_b < lo_a)
    print(f"\n{'='*70}\nSignificance check\n{'='*70}")
    print(f"{result_a['model']} 95% CI: [{lo_a:.4f}, {hi_a:.4f}]")
    print(f"{result_b['model']} 95% CI: [{lo_b:.4f}, {hi_b:.4f}]")
    if overlap:
        print("CIs OVERLAP — the accuracy difference is NOT clearly statistically "
              "significant from this test alone. Report both numbers with their CIs "
              "rather than claiming one model is definitively better.")
    else:
        print("CIs DO NOT overlap — the accuracy difference looks statistically "
              "meaningful, but remember this is on two DIFFERENT, differently-sized "
              "test sets (see the caveat below), not a paired test on identical samples.")


def print_results_table(results):
    print(f"\n{'='*70}\nSummary table (paste into your paper, reformat as needed)\n{'='*70}")
    header = f"{'Model':<10}{'N':>8}{'Accuracy':>12}{'95% CI':>18}{'Macro F1':>12}{'Latency(ms)':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        ci_str = f"[{r['accuracy_95ci'][0]:.3f},{r['accuracy_95ci'][1]:.3f}]"
        print(f"{r['model']:<10}{r['n_test_samples']:>8}{r['accuracy']:>12.4f}"
              f"{ci_str:>18}{r['macro_f1']:>12.4f}{r['avg_latency_ms']:>14.2f}")
def main():

    results = []

    # ---------------- STGCN ----------------

    from net.st_gcn import Model as STGCNModel

    print("Loading ST-GCN dataset...")

    X_stgcn, y_stgcn = load_stgcn_npy_dataset()

    print("Total ST-GCN samples:", len(X_stgcn))

    _, X_stgcn_test, _, y_stgcn_test = train_test_split(
        X_stgcn,
        y_stgcn,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_stgcn
    )

    print("ST-GCN test samples:", len(X_stgcn_test))

    stgcn_model = STGCNModel(
        in_channels=3,
        num_class=3,
        graph_args={
            "layout": "mediapipe",
            "strategy": "uniform"
        },
        edge_importance_weighting=True
    )

    stgcn_model.load_state_dict(
        torch.load(
            STGCN_MODEL_PATH,
            map_location="cpu"
        )
    )

    stgcn_model.eval()

    print("ST-GCN model loaded successfully")

    def stgcn_predict(window):
        with torch.no_grad():
            x = np.expand_dims(
                window,
                axis=-1
            )

        # (3,120,33,1) -> (1,3,120,33,1)
            x = np.expand_dims(
                x,
                axis=0
            )

            x = torch.tensor(
                x,
                dtype=torch.float32
            )

            probs = torch.softmax(
                stgcn_model(x),
                dim=1
            )

            return int(
                torch.argmax(
                    probs,
                    dim=1
                ).item()
            )

    print("Running ST-GCN inference...")

    preds, latencies = run_inference(
        stgcn_predict,
        X_stgcn_test
    )

    results.append(
        summarize(
            "STGCN",
            y_stgcn_test,
            preds,
            latencies
        )
    )

    print_results_table(results)


if __name__ == "__main__":
    main()

