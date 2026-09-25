import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

        # Same preprocessing used during training
        # (120, 33, 3) -> (3, 120, 33)
        

        # (3, 120, 33) -> (3, 120, 33, 1)
        x = np.expand_dims(x, axis=-1)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


# ---------------- LOAD DATA ---------------- #

dataset = PostureDataset()

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

print("Total samples:", len(dataset))
print("Test samples:", len(test_dataset))


# ---------------- LOAD MODEL ---------------- #

model = Model(
    in_channels=3,
    num_class=3,
    graph_args={
        "layout": "mediapipe",
        "strategy": "uniform"
    },
    edge_importance_weighting=True
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/stgcn_best.pth",
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully")


# ---------------- TEST MODEL ---------------- #

true_labels = []
predicted_labels = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(DEVICE)

        output = model(x)

        predictions = torch.argmax(
            output,
            dim=1
        )

        true_labels.extend(
            y.numpy()
        )

        predicted_labels.extend(
            predictions.cpu().numpy()
        )


# ---------------- METRICS ---------------- #

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)


print("\n========== ST-GCN TEST RESULTS ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=["good", "moderate", "bad"],
        zero_division=0
    )
)

print("========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        true_labels,
        predicted_labels
    )
)