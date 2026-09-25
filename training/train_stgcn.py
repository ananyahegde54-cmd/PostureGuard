import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from net.st_gcn import Model


# ---------------- SETTINGS ---------------- #

DATA_PATH = "data/stgcn_dataset"

CLASSES = {
    "good": 0,
    "moderate": 1,
    "bad": 2
}

CLASS_NAMES = ["Good", "Moderate", "Bad"]

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------- DATASET ---------------- #

class PostureDataset(Dataset):

    def __init__(self):

        self.samples = []
        self.labels = []

        for class_name, label in CLASSES.items():

            folder = os.path.join(
                DATA_PATH,
                class_name
            )

            for file in os.listdir(folder):

                if file.endswith(".npy"):

                    path = os.path.join(
                        folder,
                        file
                    )

                    data = np.load(path)

                    self.samples.append(data)
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        x = self.samples[index]
        y = self.labels[index]

        # Already: (3,120,33)
        # Add person dimension:
        # (3,120,33) -> (3,120,33,1)

        x = np.expand_dims(
            x,
            axis=-1
        )

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        y = torch.tensor(
            y,
            dtype=torch.long
        )

        return x, y


# ---------------- LOAD DATA ---------------- #

dataset = PostureDataset()

print("Total samples:", len(dataset))

indices = np.arange(
    len(dataset)
)

labels = np.array(
    dataset.labels
)

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_dataset = torch.utils.data.Subset(
    dataset,
    train_idx
)

test_dataset = torch.utils.data.Subset(
    dataset,
    test_idx
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))


# ---------------- MODEL ---------------- #

model = Model(
    in_channels=3,
    num_class=3,
    graph_args={
        "layout": "mediapipe",
        "strategy": "uniform"
    },
    edge_importance_weighting=True
).to(DEVICE)


# ---------------- LOSS & OPTIMIZER ---------------- #

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ---------------- TRAINING ---------------- #

best_accuracy = 0.0

print("\n================================")
print("       ST-GCN TRAINING")
print("================================")

print("Device:", DEVICE)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(
            output,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # ---------------- VALIDATION ---------------- #

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            output = model(x)

            pred = torch.argmax(
                output,
                dim=1
            )

            y_true.extend(
                y.cpu().numpy()
            )

            y_pred.extend(
                pred.cpu().numpy()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    # ---------------- SAVE BEST MODEL ---------------- #

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "models/stgcn_best.pth"
        )

        print(
            f"  ✓ Best model saved "
            f"({best_accuracy * 100:.2f}%)"
        )


# ---------------- FINAL RESULTS ---------------- #

print("\n================================")
print("       FINAL ST-GCN RESULTS")
print("================================")

print(
    f"\nBest Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)

print("Confusion Matrix:\n")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)

print("\n================================")
print("Training completed.")
print("Best model saved to:")
print("models/stgcn_best.pth")
print("================================")