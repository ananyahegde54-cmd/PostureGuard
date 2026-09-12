import os
import numpy as np
import torch

from net.st_gcn import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(
    in_channels=3,
    num_class=3,
    graph_args={
        "layout": "mediapipe",
        "strategy": "uniform"
    },
    edge_importance_weighting=True
).to(DEVICE)

model.load_state_dict(torch.load("models/stgcn_best.pth", map_location=DEVICE))
model.eval()

labels = {
    "good":0,
    "moderate":1,
    "bad":2
}

for cls in labels:

    folder = f"data/new_data/{cls}"

    file = sorted(os.listdir(folder))[0]

    x = np.load(os.path.join(folder,file))

    x = x - x.mean(axis=0, keepdims=True)

    x = np.transpose(x,(2,0,1))
    x = np.expand_dims(x,-1)
    x = np.expand_dims(x,0)

    x = torch.tensor(x,dtype=torch.float32).to(DEVICE)

    with torch.no_grad():

        out=model(x)

        prob=torch.softmax(out,dim=1)

        print(cls)
        print(prob)