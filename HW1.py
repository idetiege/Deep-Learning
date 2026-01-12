import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np


class VanillaNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
                nn.Linear(13, 24),
                nn.ReLU(),
                nn.Linear(24, 36),
                nn.ReLU(),
                nn.Linear(36, 1)
            )
        
    def forward(self, x):
        return self.network(x)


def load_data(batch_size):

    x_list = []
    y_list = []

    with open("auto-mpg.data", 'r') as f:
        for line in f:
            parts = line.split()
            if parts[3] == "?":
                continue

            vals = list(map(float, parts[:8]))
            mpg = vals[0]
            features = vals[1:8]

            x_list.append(features)
            y_list.append(mpg)

            X = torch.tensor(x_list, dtype=torch.float32)
            y = torch.tensor(y_list, dtype=torch.float32).reshape(-1, 1)

            dataset = TensorDataset(X, y)

            n_train = int(0.8 * len(dataset))
            n_test = len(dataset) - n_train

            train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])

            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            return train_dl, test_dl


def train(dataloader, model, loss_fn, optimizer):

    model.train()

    num_batches = len(dataloader)
    train_loss= 0

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
    print(f"Test loss: {train_loss:>8f} \n")

    return train_loss


def test(dataloader, model, loss_fn):

    model.eval()

    num_batches = len(dataloader)
    test_loss= 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    print(f"Test loss: {test_loss:>8f} \n")

    return test_loss


if __name__ == "__main__":


    model = VanillaNetwork()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 64

    train_dl, test_dl = load_data(batch_size)

    epochs = 1000
    train_losses = []
    test_losses = []

    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        train_loss = train(train_dl, model, loss_fn, optimizer)
        test_loss = test(test_dl, model, loss_fn)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plt.figure()
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()