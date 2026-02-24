import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()

        self.activation = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(7, 30),
            self.activation,
            nn.Linear(30, 30),
            self.activation,
            nn.Linear(30, 60),
        )

    def forward(self, x):
        return self.network(x)

class Network(nn.Module):
    def __init__(self, K0):
        super(Network, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(60, 60), bias = False,
        )

        # self.k = nn.Parameter(K0, requires_grad=True)

class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()

        self.activation = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(60, 30),
            self.activation,
            nn.Linear(30, 30),
            self.activation,
            nn.Linear(30, 7)
        )

    def forward(self, x):
        return self.network(x)
    
def Loss_recon(y_true):
    

if __name__ == "__main__":

    ntraj = 2148  # number of trajectories
    nt = 50  # number of time steps
    ny = 7  # number of states

    tvec = np.linspace(0, 350, nt)
    Y = np.loadtxt('kdata.txt').reshape(ntraj, nt, ny)
    Ytrain = Y[:2048, :, :]  # 2048 training trajectories
    Ytest = Y[2048:, :, :]  # 100 testing trajectoreis

    plt.plot(tvec, Ytrain[0, :, 0], label='y1', linestyle='dashed')
    plt.plot(tvec, Ytrain[0, :, 1], label='y2', linestyle='dashed')
    plt.plot(tvec, Ytrain[0, :, 2], label='y3', linestyle='dashed')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('state')
    plt.show()

    lr = 1e-3
    optimizer = Adam(model.parameters(), lr=lr)