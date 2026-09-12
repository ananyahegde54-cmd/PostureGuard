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

DATA_PATH = "data/new_data"

CLASSES = {
    "good": 0,
    "moderate": 1,
    "bad": 2
}

CLASS_NAMES = ["Good", "Moderate", "Bad"]

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------- DATASET ---------------- #

class PostureDataset(Dataset):

    def __init__(self):

        self.samples = []
        self.labels = []

        for class_name, label in CLASSES.items():

            folder = os.path.join(DATA_PATH, class_name)

            for file in os.listdir(folder):

                if file.endswith(".npy"):

                    path = os.path.join(folder, file)

                    data = np.load(path)

                    self.samples.append(data)
                    self.labels.append(label)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        x = self.samples[index]
        y = self.labels[index]

        # (120,33,3) -> (3,120,33)
        x = np.transpose(x, (2, 0, 1))

        # (3,120,33) -> (3,120,33,1)
        x = np.expand_dims(x, axis=-1)

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

indices = np.arange(len(dataset))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=dataset.labels
)

test_dataset = torch.utils.data.Subset(
    dataset,
    test_idx
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test samples:", len(test_dataset))


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


# ---------------- LOAD TRAINED MODEL ---------------- #

model.load_state_dict(
    torch.load(
        "models/stgcn_best.pth",
        map_location=DEVICE
    )
)

model.eval()

print("\nModel loaded successfully")


# ---------------- TEST MODEL ---------------- #

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


# ---------------- RESULTS ---------------- #

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n================================")
print("       ST-GCN TEST RESULTS")
print("================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
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