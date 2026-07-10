import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ── 1. Load & inspect data ────────────────────────────────
df = pd.read_csv("posture_data.csv")
print(f"Total rows: {len(df)}")
print("Label distribution:")
for lbl, name in {0: "Good", 1: "Moderate", 2: "Bad"}.items():
    n = (df["label"] == lbl).sum()
    print(f"  {lbl} ({name:<10}): {n:>5} rows  ({100*n/len(df):.1f}%)")

X = df.drop("label", axis=1).values.astype(np.float32)  # (N, 99)
y = df["label"].values.astype(np.int64)                  # (N,)  0/1/2

# ── 2. Build sequences of 30 consecutive frames ───────────
SEQ_LEN = 30

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])   # label of LAST frame in window
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

print("\nCreating sequences...")
X_seq, y_seq = make_sequences(X, y, SEQ_LEN)
print(f"  Sequence tensor:  {X_seq.shape}")   # (N-30, 30, 99)
print(f"  Sequence labels:  {np.bincount(y_seq)}")

# ── 3. Stratified train/test split ────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)
print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

# ── 4. Class-balanced loss weights ────────────────────────
# Critical: moderate posture is hardest to collect consistently,
# so we weight classes inversely proportional to frequency.
cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
weights = torch.tensor(cw, dtype=torch.float32)
print(f"Class weights  →  Good: {cw[0]:.3f}  Moderate: {cw[1]:.3f}  Bad: {cw[2]:.3f}")

# ── 5. Dataset ────────────────────────────────────────────
class PostureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)    # Long — needed for CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(PostureDataset(X_train, y_train),
                          batch_size=32, shuffle=True, drop_last=True)
test_loader  = DataLoader(PostureDataset(X_test,  y_test),
                          batch_size=64, shuffle=False)

# ── 6. LSTM Model — 3-class output ───────────────────────
class PostureLSTM(nn.Module):
    """
    Input:  (batch, seq_len=30, features=99)
    Output: (batch, 3)  raw logits for [Good, Moderate, Bad]

    CrossEntropyLoss applies softmax internally — do NOT add
    a sigmoid or softmax layer here during training.
    Use torch.softmax(logits, dim=1) at inference time.
    """
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
        out = self.norm(out[:, -1, :])   # last time step + layer norm
        return self.fc(self.drop(out))   # raw logits

model = PostureLSTM()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")

# ── 7. Training config ────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=25, eta_min=1e-5
)
EPOCHS = 25

# ── 8. Training loop ──────────────────────────────────────
print("\nStarting training (3-class LSTM)...\n")
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss    += loss.item()
        preds          = logits.argmax(dim=1)
        train_correct += (preds == yb).sum().item()
        train_total   += yb.size(0)

    scheduler.step()
    train_acc = 100 * train_correct / train_total

    # Validate
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds      = model(xb).argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total   += yb.size(0)
    val_acc = 100 * val_correct / val_total

    # Save best
    tag = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "posture_model.pth")
        tag = "  ← saved"

    print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
          f"loss {train_loss/len(train_loader):.4f}  "
          f"train {train_acc:.1f}%  val {val_acc:.1f}%{tag}")

# ── 9. Final evaluation ───────────────────────────────────
print(f"\nBest val accuracy: {best_val_acc:.1f}%")
model.load_state_dict(torch.load("posture_model.pth", map_location="cpu"))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_preds.extend(model(xb).argmax(dim=1).tolist())
        all_true.extend(yb.tolist())

print("\nClassification Report:")
print(classification_report(
    all_true, all_preds,
    target_names=["Good", "Moderate", "Bad"],
    digits=3
))

print("Confusion Matrix (rows=true, cols=predicted):")
cm = confusion_matrix(all_true, all_preds)
print(f"             Good  Moderate  Bad")
for i, row_name in enumerate(["Good    ", "Moderate", "Bad     "]):
    print(f"  {row_name}  {cm[i]}")

print("\nModel saved as posture_model.pth")
print("Training complete!")
