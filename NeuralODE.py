import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy
from torchdiffeq import odeint

# odeint(func, y0, t)

class NeuODE(nn.Module):
    def __init__(self, width):
        super(NeuODE, self).__init__()

        self.activation = nn.SiLU()

        self.network = nn.Sequential(
            nn.Linear(2, width), # Inputs: y0, y1
            self.activation,
            nn.Linear(width, width),
            self.activation,
            nn.Linear(width, width),
            self.activation,
            nn.Linear(width, 2) # Outputs: dy0/dt, dy1/dt
        )

        # layers = []
        # layers.append(nn.Linear(2, width))
        # layers.append(self.activation)
        # for _ in range(hlayers-1):
        #     layers.append(nn.Linear(width, width))
        #     layers.append(self.activation)

    def odefunc(self, t, y): # Output dydt
        return self.network(y)

    def forward(self, y0, tsteps):
        return odeint(self.odefunc, y0, tsteps) # 

def data():
    # Input data
    data = np.loadtxt("in_class_data.txt")

    # Move data to tansors that can be use on the GPU
    t_train = torch.tensor(data[:, 0], dtype=torch.float64) # nt
    y_train = torch.tensor(data[:, 1:], dtype=torch.float64) # nt x 2
    print(y_train.shape)


    return t_train, y_train


def train(y_train, t_train, model, optimizer, lossfn):

    model.train()
    optimizer.zero_grad()

    yhat = model(y_train[0, :], t_train)
    loss = lossfn(yhat, y_train)

    loss.backward()
    optimizer.step()

    return loss.item()

if __name__ == "__main__":

    d = data()
    t_train = d[0]
    y_train = d[1]

    nstates = 2
    width = 20
    epochs = 200
    model = NeuODE(width=width).double()
    y0 = torch.tensor(d[1][0], dtype = torch.float64)

    
    lossfn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        loss = train(y_train, t_train, model, optimizer, lossfn)
        print(f"Loss: {loss:.4f}")

        with torch.no_grad():
            yhat = model(y0, t_train)

        if _ % 20 == 0:
            plt.plot(t_train, y_train[:, 0], label="Rabbits", color="lightblue")
            plt.plot(t_train, yhat[:, 0].detach(), label="Rabbits (Predicted)", linestyle="dashed", color="lightblue")
            plt.xlabel("time")
            plt.ylabel("Population")
            plt.plot(t_train, y_train[:, 1], label="Wolves", color="orange")
            plt.plot(t_train, yhat[:, 1].detach(), label="Wolves (Predicted)", linestyle="dashed", color="orange")
            plt.legend()
            plt.show()