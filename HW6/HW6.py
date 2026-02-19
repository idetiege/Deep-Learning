import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from torch.utils.data import DataLoader, TensorDataset

class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()

        self.activation = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(128, 256),
            self.activation,
            nn.Linear(256, 256),
            self.activation,
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.network(x)

class NeuODE(nn.Module):
    def __init__(self):
        super(NeuODE, self).__init__()

        self.activation = nn.SiLU()

        self.network = nn.Sequential(
            nn.Linear(2, 128),
            self.activation,
            nn.Linear(128, 2),
        )

    def odefunc(self, t, y): # Output dydt
        return self.network(y)

    def forward(self, y0, tsteps):
        return odeint(self.odefunc, y0, tsteps)

class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()

        self.activation = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(2, 256),
            self.activation,
            nn.Linear(256, 256),
            self.activation,
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.network(x)

##############################
## Data Stuff
def generatetrajectories(ntraj, tsteps, A, trainflag):

    nx, nz = A.shape
    nt = len(tsteps)

    if trainflag:
        z1 = np.random.uniform(low=-1.5, high=0.5, size=ntraj)
        z2 = np.random.uniform(low=-1, high=1, size=ntraj)
    else:
        z1 = np.random.uniform(low=-1.5, high=1.5, size=ntraj)
        z2 = np.random.uniform(low=-1, high=1, size=ntraj)
    Z0 = np.column_stack((z1, z2))  # ntraj x nz

    Z = np.zeros((ntraj, nt, nz))

    def zode(t, z):
        return [z[1], z[0]-z[0]**3]

    for i in range(ntraj):
        sol = solve_ivp(zode, (tsteps[0], tsteps[-1]), Z0[i, :], t_eval=tsteps)
        Z[i, :, :] = sol.y.T

    # map to high dimensional space
    X = np.zeros((ntraj, nt, nx))
    for i in range(nt):
        X[:, i, :] = Z[:, i, :]**3 @ A.T

    return X

def getdata(ntrain, ntest, ncol, t_train, t_test):

    nz = 2
    nx = 128

    A = np.random.normal(size=(nx, nz))

    Xtrain = generatetrajectories(ntrain, t_train, A, trainflag=True)
    Xtest = generatetrajectories(ntest, t_test, A, trainflag=False)

    # collocation points
    z1 = np.random.uniform(low=0.5, high=1.5, size=ncol)
    z2 = np.random.uniform(low=-1, high=1, size=ncol)
    Zcol = np.column_stack((z1, z2))  # ncol x nz
    hZ = np.column_stack((Zcol[:, 1], Zcol[:, 0] - Zcol[:, 0]**3))
    fcol = np.zeros((ncol, nx))
    for i in range(ncol):
        fcol[i, :] =  hZ[[i], :] @ (3 * A * Zcol[i, :].T**2).T
    Xcol = Zcol**3 @ A.T

    return Xtrain, Xtest, Xcol, fcol, A

def true_encoder(X, A):  # X is npts * nt * nx
    Z3 = X @ np.linalg.pinv(A).T  # pinv is nz x nx
    return np.sign(Z3) * np.abs(Z3)**(1/3)

def data(trainData, testData, batch_size):

    # Normalize
    mean_train = np.mean(trainData)
    stddev_train = np.std(trainData)

    mean_test = np.mean(trainData)
    stdev_test = np.std(testData)

    train_norm = (trainData - mean_train) / stddev_train
    test_norm = (testData - mean_test) / stdev_test

    trainDl = DataLoader(train_norm, batch_size, shuffle = True)
    testDl = DataLoader(test_norm, batch_size, shuffle = True)
    return trainDl, testDl

##############################

## Loss Functions

def Loss_data(loss_fn, x, encoder, decoder, ode, x_batch, t_batch):

    z = encoder(x_batch)
    x_recon = decoder(z)
    lossRecon = loss_fn(x_recon, x_batch)

    z0 = z[:, 0, :]
    z_hat = ode(z0, t_batch)
    z_hat = z_hat.permute(1, 0, 2) # nt x ntraj x nz

    x_hat = decoder(z_hat)

    lossPred = loss_fn(x_hat, x_batch)

    return lossPred, lossRecon

def Loss_phys(loss_fn, f, Xcol, x_coll, z, ode, encoder, decoder):
    
    dzdx = torch.zeros(Xcol.shape[0], 2, Xcol.shape[2], device=Xcol.device, dtype=Xcol.dtype)
    dzdx[:, 0, :] = torch.autograd.grad(z[:, 0], x_coll, grad_outputs=torch.ones_like(z[:,0]), retain_graph=True,  create_graph=True)[0]
    dzdx[:, 1, :] = torch.autograd.grad(z[:, 1], x_coll, grad_outputs=torch.ones_like(z[:,0]), retain_graph=True,  create_graph=True)[0]

    dzdt = torch.bmm(dzdx, f.unsqueeze(2)).squeeze() # ncol x 2

    zdot = ode.odefunc(None, z)

    lossPred = loss_fn(dzdt, zdot)
    lossColl = loss_fn(Xcol, decoder(encoder(Xcol)))

    return lossPred + lossColl

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    print("device:", device)


    # discretization in time for training and test data.  These don't need to be changed.
    nt_train = 11
    nt_test = 21
    t_train = np.linspace(0.0, 1.0, nt_train)
    t_test = np.linspace(0.0, 1.0, nt_test)

    # number of training pts, testing pts, and collocation pts.
    # You will need more training pts and collocation pts eventually (testing pts can remain as is).
    ntrain = 100
    ntest = 100
    ncol = 1000
    Xtrain, Xtest, Xcol, fcol, Amap = getdata(ntrain, ntest, ncol, t_train, t_test)

    # Xtrain is ntrain x nt_train x nx
    # Xtest is ntest x nt_test x nx
    # Xcol is ncol x nx
    # fcol is ncol x nx and represents f(Xcol)
    # Amap is only needed for final plot (see function below)



    # once you have a prediction for Xhat(t) (ntest x nt_test x nx)
    # this will use this specific projection to Z, to create a plot
    # like the bottom right corner of Fig 3
    Zhat = true_encoder(Xhat, Amap)

    plt.figure()
    for i in range(0, ntest):
        plt.plot(Zhat[i, 0, 0], Zhat[i, 0, 1], "ko")
        plt.plot(Zhat[i, :, 0], Zhat[i, :, 1], "k")
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1, 1])

    plt.show()