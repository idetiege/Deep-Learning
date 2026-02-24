import numpy as np
import torch
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
            nn.Linear(60, 60, bias=False), 
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

def data():
    ntraj = 2148  # number of trajectories
    nt = 50  # number of time steps
    ny = 7  # number of states

    tvec = np.linspace(0, 350, nt)
    Y = np.loadtxt('kdata.txt').reshape(ntraj, nt, ny)
    Ytrain = Y[:2048, :, :]  # 2048 training trajectories
    Ytest = Y[2048:, :, :]  # 100 testing trajectoreis

    Ytrain_torch = torch.from_numpy(Ytrain).float() # Size: (2048, 50, 7)

    y_t = Ytrain_torch[:, :-1, :] # (N_traj, N_t-1, 7)
    y_t1 = Ytrain_torch[:, 1:, :] # (N_traj, N_t-1, 7)

    y_t_flat = y_t.reshape(-1, 7) # (N_traj*(N_t-1), 7)
    y_t1_flat = y_t1.reshape(-1, 7) # (N_traj*(N_t-1), 7)

    return tvec, Ytrain, Ytest, y_t_flat, y_t1_flat

def Loss_recon(loss_fn, encoder, decoder, y_batch):

    z = encoder(y_batch)
    Y_hat = decoder(z)

    L = loss_fn(Y_hat, y_batch)
    return L

def Loss_linear(loss_fn,encoder, K, y_t, y_t1):
    z_t = encoder(y_t)
    z_t1 = encoder(y_t1)
    z_t1_hat =  K(z_t)

    L = loss_fn(z_t1, z_t1_hat)
    return L

def Loss_pred(loss_fn, encoder, decoder, network, y_t_flat):
    
    z_t = encoder(y_t_flat)
    z_t1_hat = network(z_t)
    y_t1_hat = decoder(z_t1_hat)
    
    L = loss_fn(y_t1_hat, y_t1_flat)
    return L



if __name__ == "__main__":

    tvec, Ytrain, Ytest, y_t, y_t1, y_t1_flat = data()

    plt.plot(tvec, Ytrain[0, :, 0], label='y1', linestyle='dashed')
    plt.plot(tvec, Ytrain[0, :, 1], label='y2', linestyle='dashed')
    plt.plot(tvec, Ytrain[0, :, 2], label='y3', linestyle='dashed')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('state')
    plt.show()

    lr = 1e-3
    optimizer = Adam(model.parameters(), lr=lr)