import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True  # small speed boost on GPU

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()

        self.activation = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(7, 14),
            self.activation,
            nn.Linear(14, 14),
            self.activation,
            nn.Linear(14, 21),
        )

    def forward(self, x):
        return self.network(x)


class Koopman(nn.Module):
    def __init__(self):
        super(Koopman, self).__init__()

        self.K = nn.Sequential(
            nn.Linear(21, 21, bias=False),
        )

    def forward(self, x):
        return self.K(x)


class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()

        self.activation = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(21, 14),
            self.activation,
            nn.Linear(14, 14),
            self.activation,
            nn.Linear(14, 7),
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------
# Data utilities
# ---------------------------------------------------------

def load_data(kdata_path="kdata.txt"):
    """
    Load glycolysis data, split train/test, normalize using
    training stats, and build one-step pairs (y_t, y_{t+1}).
    """
    ntraj = 2148
    nt = 50
    ny = 7

    tvec = np.linspace(0.0, 350.0, nt)

    Y = np.loadtxt(kdata_path).reshape(ntraj, nt, ny)
    Ytrain = Y[:2048, :, :]   # (N_train, T, 7)
    Ytest  = Y[2048:, :, :]   # (N_test,  T, 7)

    # normalize by training stats
    mu  = Ytrain.mean(axis=(0, 1), keepdims=True)        # (1,1,7)
    std = Ytrain.std(axis=(0, 1), keepdims=True) + 1e-8  # (1,1,7)

    Ytrain_norm = (Ytrain - mu) / std
    Ytest_norm  = (Ytest  - mu) / std

    # one-step pairs from normalized training data
    Ytrain_torch = torch.from_numpy(Ytrain_norm).float()  # (N, T, 7)

    y_t  = Ytrain_torch[:, :-1, :]   # (N, T-1, 7)
    y_t1 = Ytrain_torch[:, 1:,  :]   # (N, T-1, 7)

    y_t_flat  = y_t.reshape(-1, ny).numpy()   # (N*(T-1), 7)
    y_t1_flat = y_t1.reshape(-1, ny).numpy()

    # a few sample TRAIN trajectories (normalized) if you want them
    first_train_traj  = Ytrain_norm[0]   # (T,7)
    second_train_traj = Ytrain_norm[3]
    third_train_traj  = Ytrain_norm[6]

    return (tvec,
            Ytrain_norm, Ytest_norm,
            y_t_flat, y_t1_flat,
            first_train_traj, second_train_traj, third_train_traj,
            mu, std)


# ---------------------------------------------------------
# Losses
# ---------------------------------------------------------

def loss_recon(loss_fn, encoder, decoder, y_batch):
    z = encoder(y_batch)
    y_hat = decoder(z)
    return loss_fn(y_hat, y_batch)


def loss_linear(loss_fn, encoder, K, y_t, y_t1):
    z_t  = encoder(y_t)
    z_t1 = encoder(y_t1)
    z_t1_hat = K(z_t)
    return loss_fn(z_t1_hat, z_t1)


def loss_pred(loss_fn, encoder, decoder, koopman, y_t, y_t1):
    z_t      = encoder(y_t)
    z_t1_hat = koopman(z_t)
    y_t1_hat = decoder(z_t1_hat)
    return loss_fn(y_t1_hat, y_t1)


def loss_multistep(loss_fn, encoder, decoder, koopman, Ytraj, H=15):
    """
    Multi-step rollout loss over a batch of trajectories.

    Ytraj : (Ntraj, T, 7) normalized trajectories (torch, on device)
    H     : prediction horizon in steps
    """
    N, T, ny = Ytraj.shape
    H = min(H, T - 1)

    y0 = Ytraj[:, 0, :]   # (N, 7)
    z  = encoder(y0)      # (N, 21)

    loss = 0.0
    for h in range(1, H + 1):
        z      = koopman(z)                # advance latent
        y_hat  = decoder(z)                # (N, 7)
        y_true = Ytraj[:, h, :]            # (N, 7)
        loss   = loss + loss_fn(y_hat, y_true)

    return loss / H


# ---------------------------------------------------------
# Training / testing / rollout
# ---------------------------------------------------------

def train_epoch(encoder, decoder, koopman, optimizer, loss_fn,
                loader, Ytrain_t, device,
                w_pred=1.0, w_linear=1.0, w_recon=0.1, w_multi=1.0,
                H=15, n_multi_traj=256):
    """
    One epoch:
      - loop over batches for one-step losses
      - one extra step for multi-step loss on a subset of trajectories
    """
    encoder.train(); decoder.train(); koopman.train()

    epoch_loss = 0.0
    n_batches = 0

    # ---- batch one-step training ----
    for y_t_batch, y_t1_batch in loader:
        y_t_batch  = y_t_batch.to(device)
        y_t1_batch = y_t1_batch.to(device)

        optimizer.zero_grad()

        L_pred   = loss_pred(loss_fn, encoder, decoder, koopman, y_t_batch, y_t1_batch)
        L_linear = loss_linear(loss_fn, encoder, koopman, y_t_batch, y_t1_batch)
        L_recon  = loss_recon(loss_fn, encoder, decoder, y_t_batch)

        loss = (w_pred   * L_pred +
                w_linear * L_linear +
                w_recon  * L_recon)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    # ---- one multi-step update per epoch (on random subset of trajectories) ----
    Ntraj = Ytrain_t.size(0)
    n_multi_traj = min(n_multi_traj, Ntraj)

    idx = torch.randperm(Ntraj, device=device)[:n_multi_traj]
    Y_multi = Ytrain_t[idx]   # (n_multi_traj, T, 7)

    optimizer.zero_grad()
    L_multi = loss_multistep(loss_fn, encoder, decoder, koopman, Y_multi, H=H)
    (w_multi * L_multi).backward()
    optimizer.step()

    avg_loss = epoch_loss / n_batches
    return avg_loss, L_multi.item()


def eval_full(encoder, decoder, koopman, loss_fn, y_t_flat, y_t1_flat, device):
    encoder.eval(); decoder.eval(); koopman.eval()

    with torch.no_grad():
        y_t  = y_t_flat.to(device)
        y_t1 = y_t1_flat.to(device)

        L_pred   = loss_pred(loss_fn, encoder, decoder, koopman, y_t, y_t1)
        L_linear = loss_linear(loss_fn, encoder, koopman, y_t, y_t1)
        L_recon  = loss_recon(loss_fn, encoder, decoder, y_t)

        loss = L_pred + L_linear + L_recon

    return loss.item()


def rollout_prediction(encoder, decoder, koopman, traj_norm, device, mu, std):
    """
    Roll out a single trajectory from its first (normalized) state
    using learned Koopman dynamics, then return *denormalized* states.
    """
    encoder.eval(); decoder.eval(); koopman.eval()

    nt = traj_norm.shape[0]

    y0 = torch.tensor(traj_norm[0], dtype=torch.float32, device=device).unsqueeze(0)  # (1,7)
    z  = encoder(y0)

    preds = []

    with torch.no_grad():
        for _ in range(nt):
            y_hat = decoder(z)               # (1,7)
            preds.append(y_hat.squeeze(0).cpu().numpy())
            z = koopman(z)

    preds = np.array(preds)   # (nt, 7)

    # denormalize
    preds_denorm = preds * std[0, 0, :] + mu[0, 0, :]

    return preds_denorm


# ---------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------

def plot_train_test(train_losses, test_losses, savepath=None):
    train_losses = np.asarray(train_losses, dtype=float)
    test_losses  = np.asarray(test_losses,  dtype=float)

    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses,  "--", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Test Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200)
    else:
        plt.show()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32
    print("device:", device)

    (tvec,
     Ytrain_norm, Ytest_norm,
     y_t_flat, y_t1_flat,
     first_train_traj, second_train_traj, third_train_traj,
     mu, std) = load_data("/content/kdata.txt")

    nt = Ytrain_norm.shape[1]

    # tensors for loader and eval
    y_t_flat_t  = torch.tensor(y_t_flat,  dtype=dtype, device=device)
    y_t1_flat_t = torch.tensor(y_t1_flat, dtype=dtype, device=device)
    Ytrain_t    = torch.tensor(Ytrain_norm, dtype=dtype, device=device)

    dataset = TensorDataset(y_t_flat_t, y_t1_flat_t)
    # larger batch size to better use GPU
    train_loader = DataLoader(dataset, batch_size=8192, shuffle=True)

    # models
    encoder = Encode().to(device).to(dtype)
    decoder = Decode().to(device).to(dtype)
    koopman = Koopman().to(device).to(dtype)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(koopman.parameters())
    optimizer = AdamW(params, lr=1e-3)
    loss_fn = nn.MSELoss()

    # loss weights
    w_pred   = 1.0
    w_linear = 1.0
    w_recon  = 0.1
    w_multi  = 1.0
    H        = 15
    n_multi_traj = 256   # trajectories used for multi-step loss each epoch

    n_epochs = 1000
    train_losses = []
    test_losses  = []

    for epoch in range(n_epochs):
        train_loss, L_multi_val = train_epoch(
            encoder, decoder, koopman, optimizer, loss_fn,
            train_loader, Ytrain_t, device,
            w_pred=w_pred, w_linear=w_linear,
            w_recon=w_recon, w_multi=w_multi,
            H=H, n_multi_traj=n_multi_traj
        )

        test_loss  = eval_full(encoder, decoder, koopman, loss_fn,
                               y_t_flat_t, y_t1_flat_t, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"Train: {train_loss:.4e} | "
                  f"Test: {test_loss:.4e} | "
                  f"Multi-step loss: {L_multi_val:.4e}")

    # -------------------------------------------------
    # Rollout for FIRST TEST trajectory (assignment)
    # -------------------------------------------------
    first_test_traj_norm = Ytest_norm[0]              # (T,7), normalized

    pred_test = rollout_prediction(
        encoder,
        decoder,
        koopman,
        first_test_traj_norm,
        device,
        mu,
        std
    )

    # true test trajectory, denormalized
    true_test_traj = first_test_traj_norm * std[0, 0, :] + mu[0, 0, :]  # (T,7)

    # -------------------------------------------------
    # Assignment plot: first TEST trajectory, states 1–3
    # -------------------------------------------------
    plt.figure(figsize=(8, 5))

    state_labels = [r"$y_1$", r"$y_2$", r"$y_3$"]
    colors       = ["C0", "C1", "C2"]

    for i in range(3):
        # data (dashed)
        plt.plot(
            tvec,
            true_test_traj[:, i],
            linestyle="--",
            color=colors[i],
            label=f"Data {state_labels[i]}"
        )

        # model (solid)
        plt.plot(
            tvec,
            pred_test[:, i],
            linestyle="-",
            color=colors[i],
            label=f"Model {state_labels[i]}"
        )
  
    plt.xlabel("Time")
    plt.ylabel("State value")
    plt.title("First Test Trajectory: Koopman Model vs Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("koopman_first_test_3states.png", dpi=200)
    plt.show()

    # -------------------------------------------------
    # Loss vs epoch plot
    # -------------------------------------------------
    plot_train_test(train_losses, test_losses, savepath="loss_vs_epoch.png")