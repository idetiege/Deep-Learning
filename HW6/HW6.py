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
##############################

## Loss Functions

def Loss_data(loss_fn, encoder, decoder, ode, x_batch, t_batch):

    B, nt, nx = x_batch.shape

    x_flat = x_batch.reshape(-1, nx)
    z_flat = encoder(x_flat)
    z = z_flat.reshape(B, nt, 2)

    x_recon_flat = decoder(z_flat)
    x_recon = x_recon_flat.reshape(B, nt, nx)

    lossRecon = loss_fn(x_recon, x_batch)

    z0 = z[:, 0, :]
    z_hat = ode(z0, t_batch)
    z_hat = z_hat.permute(1, 0, 2) # B, nt, 2

    x_hat_flat = decoder(z_hat.reshape(-1, 2))
    x_hat = x_hat_flat.reshape(B, nt, nx)

    lossPred = loss_fn(x_hat, x_batch)

    return lossPred + lossRecon

def Loss_phys(loss_fn, encoder, ode, decoder, Xcol_t, fcol_t):
    
    Xcol_t = Xcol_t.clone().detach().requires_grad_(True)

    z = encoder(Xcol_t)

    dzdx = torch.zeros(Xcol_t.shape[0], 2, Xcol_t.shape[1], device=Xcol_t.device, dtype=Xcol_t.dtype)
    dzdx[:, 0, :] = torch.autograd.grad(z[:, 0], Xcol_t, grad_outputs=torch.ones_like(z[:,0]), retain_graph=True,  create_graph=True)[0]
    dzdx[:, 1, :] = torch.autograd.grad(z[:, 1], Xcol_t, grad_outputs=torch.ones_like(z[:,1]), retain_graph=True,  create_graph=True)[0]

    dzdt = torch.bmm(dzdx, fcol_t.unsqueeze(2)).squeeze() # ncol x 2

    zdot = ode.odefunc(None, z)

    lossPred = loss_fn(dzdt, zdot)
    lossColl = loss_fn(Xcol_t, decoder(encoder(Xcol_t)))

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
    encoder = Encode().to(device).double()
    decoder = Decode().to(device).double()
    ode = NeuODE().to(device).double()

    # Setup the optimizer to train all three networks together
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(ode.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss_fn = nn.MSELoss()

    # Normalize the data and convert to tensors
    mu = Xtrain.mean(axis=(0,1), keepdims=True)
    sigma = Xtrain.std(axis=(0,1), keepdims=True) + 1e-8
    Xtrain_norm = (Xtrain - mu) / sigma
    Xtest_norm  = (Xtest  - mu) / sigma
    Xcol_norm   = (Xcol   - mu.squeeze(0)) / sigma.squeeze(0)

    Xtrain_t = torch.tensor(Xtrain_norm, dtype=dtype, device=device)
    Xtest_t  = torch.tensor(Xtest_norm,  dtype=dtype, device=device)
    Xcol_t   = torch.tensor(Xcol_norm,   dtype=dtype, device=device)
    fcol_t   = torch.tensor(fcol,        dtype=dtype, device=device)

    t_train_t = torch.tensor(t_train, dtype=dtype, device=device)
    t_test_t  = torch.tensor(t_test,  dtype=dtype, device=device)

    # Create a DataLoader for the training data
    train_dataset = TensorDataset(Xtrain_t)
    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)

    n_epochs = 500
    # Weights for losses
    w_pred, w_phys = 1, 10

    # Training loop
    for epoch in range(n_epochs):
        encoder.train(); decoder.train(); ode.train()
        total_loss = 0.0

        for (x_batch,) in train_loader:
            lossData = Loss_data(loss_fn, encoder, decoder, ode, x_batch, t_train_t)
            lossPhys = Loss_phys(loss_fn, encoder, ode, decoder, Xcol_t, fcol_t)
            loss = w_pred*lossData + w_phys*lossPhys

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}, Data Loss: {lossData.item():.4f}, Phys Loss: {lossPhys.item():.4f}")


    encoder.eval(); decoder.eval(); ode.eval()
    with torch.no_grad():
        z0_test = encoder(Xtest_t[:, 0, :])             # (ntest, 2)
        z_hat_test = ode(z0_test, t_test_t)            # (nt_test, ntest, 2)
        z_hat_test = z_hat_test.permute(1, 0, 2)       # (ntest, nt_test, 2)

        Xhat_flat = decoder(z_hat_test.reshape(-1, 2)) # (ntest*nt_test, 128)
        Xhat_norm = Xhat_flat.reshape(Xtest_t.shape)   # (ntest, nt_test, 128)

    Xhat = Xhat_norm.cpu().numpy() * sigma + mu        # un-normalize
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