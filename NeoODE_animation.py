import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

class NeuODE(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 2),
        )

    def odefunc(self, t, y):
        return self.network(y)

    def forward(self, y0, tsteps):
        return odeint(self.odefunc, y0, tsteps)

def load_data(path="in_class_data.txt", dtype=torch.float64, device="cpu"):
    data = np.loadtxt(path)
    t = torch.tensor(data[:, 0], dtype=dtype, device=device)         # [nt]
    y = torch.tensor(data[:, 1:], dtype=dtype, device=device)        # [nt,2]
    return t, y

def train_and_record(
    t_train,
    y_train,
    model,
    optimizer,
    lossfn,
    epochs=200,
    save_every=5,
    device="cpu",
):
    """
    Returns:
      key_epochs:  [n_keys]
      key_preds:   [n_keys, nt, 2]  (float32 on CPU for compactness)
      losses:      [epochs]
    """
    model.to(device)
    model.train()

    # store losses (optional but useful for title overlays later)
    losses = np.zeros(epochs, dtype=np.float32)

    # keyframe storage
    key_epochs = []
    key_preds = []

    y0 = y_train[0]  # [2]

    for ep in range(epochs):
        optimizer.zero_grad()

        yhat = model(y0, t_train)     # [nt,2]
        loss = lossfn(yhat, y_train)
        loss.backward()
        optimizer.step()

        losses[ep] = float(loss.detach().cpu())

        # save keyframe prediction every N epochs (and also epoch 0)
        if ep % save_every == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = model(y0, t_train)  # [nt,2]
            model.train()

            # move to CPU float32 for storage efficiency
            key_epochs.append(ep)
            key_preds.append(pred.detach().cpu().float().numpy())

    key_epochs = np.array(key_epochs, dtype=np.int32)
    key_preds = np.stack(key_preds, axis=0)  # [n_keys, nt, 2]
    return key_epochs, key_preds, losses

device = "cuda" if torch.cuda.is_available() else "cpu"
t_train, y_train = load_data(device=device)

model = NeuODE(width=20).double()
lossfn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

key_epochs, key_preds, losses = train_and_record(
    t_train, y_train, model, optimizer, lossfn,
    epochs=200, save_every=5, device=device
)

# Save to disk for later animation
np.savez_compressed(
    "training_rollout_keyframes.npz",
    t=t_train.detach().cpu().numpy(),
    y=y_train.detach().cpu().numpy(),
    key_epochs=key_epochs,
    key_preds=key_preds,
    losses=losses,
)

import numpy as np

def interpolate_frames(key_epochs, key_preds, frames_per_key_interval=10):
    """
    Creates smooth frames between keyframes using linear interpolation.

    Returns:
      frames: [n_frames, nt, 2]
      frame_epoch_float: [n_frames] (for labeling if you want)
    """
    frames = []
    frame_epoch_float = []

    for i in range(len(key_epochs) - 1):
        e0, e1 = key_epochs[i], key_epochs[i + 1]
        p0, p1 = key_preds[i], key_preds[i + 1]

        # include the starting keyframe
        for j in range(frames_per_key_interval):
            a = j / frames_per_key_interval
            frames.append((1 - a) * p0 + a * p1)
            frame_epoch_float.append((1 - a) * e0 + a * e1)

    # include final keyframe
    frames.append(key_preds[-1])
    frame_epoch_float.append(float(key_epochs[-1]))

    return np.stack(frames, axis=0), np.array(frame_epoch_float, dtype=np.float32)

pack = np.load("training_rollout_keyframes.npz", allow_pickle=False)
t = pack["t"]
y = pack["y"]
key_epochs = pack["key_epochs"]
key_preds = pack["key_preds"]
losses = pack["losses"]

frames, frame_epoch_float = interpolate_frames(key_epochs, key_preds, frames_per_key_interval=12)
