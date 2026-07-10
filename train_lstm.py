import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# ── 1. Load data ──────────────────────────────────────────
df = pd.read_csv('posture_data.csv')
X = df.drop('label', axis=1).values  # shape: (3335, 99)
y = df['label'].values                # shape: (3335,)

# ── 2. Create sequences of 30 frames ─────────────────────
SEQ_LEN = 30

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])
    return np.array(Xs), np.array(ys)

print("Creating sequences...")
X_seq, y_seq = make_sequences(X, y, SEQ_LEN)
print(f"Sequence shape: {X_seq.shape}")  # should be (3305, 30, 99)

# ── 3. Train/test split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 4. Dataset class ──────────────────────────────────────
class PostureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = PostureDataset(X_train, y_train)
test_ds  = PostureDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

# ── 5. LSTM Model ─────────────────────────────────────────
class PostureLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=2):
        super(PostureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).squeeze()

model = PostureLSTM()
print(f"\nModel architecture:\n{model}")

# ── 6. Training setup ─────────────────────────────────────
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 20

# ── 7. Training loop ──────────────────────────────────────
print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (output >= 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predicted = (output >= 0.5).float()
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

# ── 8. Save model ─────────────────────────────────────────
torch.save(model.state_dict(), 'posture_model.pth')
print("\nModel saved as posture_model.pth!")
print("Training complete!")


