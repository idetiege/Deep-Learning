import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np


class VanillaNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
                nn.Linear(9, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
        
    def forward(self, x):
        return self.network(x)


def load_data(batch_size):

    x_list,y_list = [], []

    with open("auto-mpg.data", 'r') as f:
        for line in f:
            parts = line.split()
            if not parts or parts[3] == "?":
                continue

            vals = list(map(float, parts[:8]))
            mpg = vals[0]

            cyl, disp, hp, wt, acc, year, origin = vals[1:8]

            o1 = 1.0 if origin == 1 else 0.0
            o2 = 1.0 if origin == 2 else 0.0
            o3 = 1.0 if origin == 3 else 0.0

            features = [cyl, disp, hp, wt, acc, year, o1, o2, o3]
            x_list.append(features)
            y_list.append(mpg)

    X = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).reshape(-1, 1)

    n = len(X)
    n_train = int(0.8 * n)
    n_test = n - n_train
    g = torch.Generator().manual_seed(42)

    full_ds = TensorDataset(X, y)
    train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=g)

    X_train = X[train_ds.indices]
    y_train = y[train_ds.indices]
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True) + 1e-8
    X = (X - X_mean) / X_std

    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_norm = (y - y_mean) / y_std

    full_ds = TensorDataset(X, y_norm)
    train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=g)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl, y_mean, y_std


def train(dataloader, model, loss_fn, optimizer):

    model.train()

    num_batches = len(dataloader)
    train_loss = 0

    for X, y in dataloader:

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    train_loss /= num_batches
    # print(f"Train loss: {train_loss:>8f} \n")

    return train_loss


def test(dataloader, model, loss_fn):

    model.eval()

    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    # print(f"Test loss: {test_loss:>8f} \n")

    return test_loss

def unnormalize(y_norm, mean, std):
    return y_norm * std + mean

if __name__ == "__main__":


    model = VanillaNetwork()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    batch_size = 64

    train_dl, test_dl, y_mean, y_std = load_data(batch_size)

    epochs = 500
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_pred_mpg = None
    best_true_mpg = None

    for e in range(epochs):
        # print(f"Epoch {e+1}\n-------------------------------")
        train_loss = train(train_dl, model, loss_fn, optimizer)
        test_loss = test(test_dl, model, loss_fn)
        rmse_mpg = (test_loss ** 0.5) * y_std.item()
        # print(f"Test RMSE (MPG): {rmse_mpg:.2f}\n")



        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss

            X_batch, y_batch = next(iter(test_dl))
            with torch.no_grad():
                pred_norm = model(X_batch)

            pred_mpg = unnormalize(pred_norm, y_mean, y_std)
            true_mpg = unnormalize(y_batch, y_mean, y_std)

            best_pred_mpg = pred_mpg
            best_true_mpg = true_mpg

    print("Best Test RMSE (MPG): {:.2f}".format((best_test_loss ** 0.5) * y_std.item()))
    plt.figure()
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()