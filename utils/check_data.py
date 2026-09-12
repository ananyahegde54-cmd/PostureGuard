import numpy as np

files = [
    "data/good.npy",
    "data/slouching.npy",
    "data/neck_bent.npy",
    "data/rounded_shoulders.npy",
    "data/leaning.npy"
]

for file in files:
    data = np.load(file)
    print(file, "->", data.shape)