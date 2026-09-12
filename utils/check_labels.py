import numpy as np

y = np.load("data/y_zenodo.npy")

unique, counts = np.unique(y, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")