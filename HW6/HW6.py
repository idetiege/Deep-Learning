import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

class NN1(nn.Module):
    def __init__(self):
        super(NN1, self).__init__()

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
    def __init__(self, width, hlayers):
        super(NeuODE, self).__init__()

        self.activation = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(2, 128),
            self.activation,
            nn.Linear(128, 2),
        )

    def odefunc(self, t, y): # Output dydt
        return self.network(y)

    def forward(self, y0, tsteps):
        return odeint(self.odefunc, y0, tsteps)

class NN2(nn.Module):
    def __init__(self):
        super(NN2, self).__init__()

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

def train_stage(y_train, model, optimizer, lossfn, epochs):
    model.train()
    stage_losses = []

    t = torch.arange(y_train.shape[0], device=y_train.device, dtype=y_train.dtype)


    for _ in range(epochs):
        optimizer.zero_grad()
        yhat = model(y_train[0, :], t) # Size: (T,4)
        loss = lossfn(yhat, y_train)
        loss.backward()
        optimizer.step()
        stage_losses.append(loss.item())

    return stage_losses

def test(y_test, t_test, model, lossfn):

    model.eval()
    with torch.no_grad():
        yhat = model(y_test[0, :], t_test)
        loss = lossfn(yhat, y_test)

    return loss.item()

def denormalize(y_norm, mean_vals, std_vals):
    # y_norm: torch tensor (T,4)
    mu = torch.tensor(mean_vals.values, dtype=y_norm.dtype, device=y_norm.device)
    sd = torch.tensor(std_vals.values, dtype=y_norm.dtype, device=y_norm.device)
    return y_norm * sd + mu


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # try float64 only if needed
    print("device:", device)

    # Get the data
    train_data, test_data, train_dates, test_dates, mean_vals, std_vals, full_dates, full_data = data()

    train_data = train_data.to(device=device, dtype=dtype)
    test_data  = test_data.to(device=device, dtype=dtype)
    full_data  = full_data.to(device=device, dtype=dtype)

    # Call the model, loss function, and optimizer
    model = NeuODE(width=20, hlayers=3).to(device=device, dtype=dtype)
    lossfn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Set up the stages of training
    stages = [4, 8, 12, 16, 20]
    epochs_per_stage = {4: 300, 8: 300, 12: 400, 16: 400, 20: 500}

    all_losses = []

    # Train in stages
    for k in stages:
        yk = train_data[:k, :]
        dk = train_dates[:k]

        stage_losses = train_stage(yk, model, optimizer, lossfn, epochs_per_stage[k])
        all_losses.extend(stage_losses)

        # predict over the same window for plotting
        with torch.no_grad():
            t = torch.arange(k, device=train_data.device, dtype=train_data.dtype)
            yhat = model(yk[0, :], t)  # (k,4)

    # plot_stage(dk, yk, yhat, stage_losses, title=f"Stage: first {k} months")
    # Observations across ALL months
    y_obs_real = denormalize(full_data, mean_vals, std_vals)

    yhat_all = forecast_all_months(model, full_data)         # (T,4)
    y_pred_real = denormalize(yhat_all, mean_vals, std_vals)

    plot_forecast(full_dates, y_obs_real, y_pred_real, split_idx=20)




if __name__ == "__main__":

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