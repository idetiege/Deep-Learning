import torch
from torch import nn
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

def load_data(batch_size):

    data = np.loadtxt("2-housing.txt")

    features = torch.tensor(data[:, :-1], dtype = torch.float32)
    targets = torch.tensor(data[:, -1].reshape(-1, 1), dtype = torch.float32)

    dataset = TensorDataset(features, targets)

    train_ds, test_ds = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)

    return train_dataloader, test_dataloader



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