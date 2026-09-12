import os

DATA_PATH = "data/new_data"

CLASSES = {
    "good": 0,
    "moderate": 1,
    "bad": 2
}

labels = []

for class_name, label in CLASSES.items():
    folder = os.path.join(DATA_PATH, class_name)

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            labels.append(label)

print(labels)
print("Total:", len(labels))
print("Good:", labels.count(0))
print("Moderate:", labels.count(1))
print("Bad:", labels.count(2))