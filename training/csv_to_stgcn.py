import os
import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "data/posture_data_final.csv"

OUTPUT_DIR = "data/stgcn_dataset"

SEQUENCE_LENGTH = 120
STRIDE = 120

# Label mapping
LABELS = {
    0: "good",
    1: "moderate",
    2: "bad"
}

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for class_name in LABELS.values():
    os.makedirs(
        os.path.join(OUTPUT_DIR, class_name),
        exist_ok=True
    )

# ============================================================
# LOAD CSV
# ============================================================

print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)

# ============================================================
# VERIFY LABELS
# ============================================================

if "label" not in df.columns:
    raise ValueError("CSV does not contain a 'label' column.")

unique_labels = sorted(df["label"].unique())

print("Labels found:", unique_labels)

for label in unique_labels:
    if label not in LABELS:
        raise ValueError(f"Unknown label found: {label}")

# ============================================================
# LANDMARK COLUMNS
# ============================================================

landmark_columns = []

for i in range(33):
    landmark_columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

missing = [
    col for col in landmark_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing landmark columns: {missing}"
    )

# ============================================================
# EXTRACT DATA
# ============================================================

landmarks = df[landmark_columns].values.astype(np.float32)
labels = df["label"].values.astype(np.int64)

print("Landmark shape:", landmarks.shape)
print("Label shape:", labels.shape)

# ============================================================
# CREATE SEQUENCES
# ============================================================

sequence_count = 0

class_counters = {
    0: 0,
    1: 0,
    2: 0
}

print("\nCreating sequences...")

for start in range(
    0,
    len(df) - SEQUENCE_LENGTH + 1,
    STRIDE
):

    end = start + SEQUENCE_LENGTH

    sequence = landmarks[start:end]
    sequence_labels = labels[start:end]

    # --------------------------------------------------------
    # Make sure all 120 frames have the same label
    # --------------------------------------------------------

    if not np.all(sequence_labels == sequence_labels[0]):
        continue

    label = int(sequence_labels[0])

    # --------------------------------------------------------
    # Reshape
    #
    # Current:
    #     (120, 33, 3)
    #
    # ST-GCN:
    #     (3, 120, 33)
    # --------------------------------------------------------

    sequence = sequence.reshape(
        SEQUENCE_LENGTH,
        33,
        3
    )

    sequence = np.transpose(
        sequence,
        (2, 0, 1)
    )

    class_name = LABELS[label]

    class_counters[label] += 1

    filename = (
        f"{class_name}_"
        f"{class_counters[label]:04d}.npy"
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        class_name,
        filename
    )

    np.save(save_path, sequence)

    sequence_count += 1

# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("ST-GCN DATASET CREATED")
print("========================================")

print("Total sequences:", sequence_count)

for label, class_name in LABELS.items():
    print(
        f"{class_name.capitalize():10s}: "
        f"{class_counters[label]}"
    )

print("\nOutput:")
print(OUTPUT_DIR)

print("\nEach sequence shape:")
print("(3, 120, 33)")