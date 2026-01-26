import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class VanillaNetwork(nn.Module):
    def __init__(self, hlayers, width):
        super(VanillaNetwork, self).__init__()

        layers = []
        layers.append(nn.Linear(1, width))
        layers.append(nn.Tanh())

        for _ in range(hlayers-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))

        self.network = nn.Sequential(*layers)
        
    def forward(self, t): # t is size 1,1
        return self.network(t)
    
def residual(model, t, params): # t size: (ncol, 1)

    # Unpack parameters
    x, t = params

    # Evaluate model
    u = model(t) # y size: (ncol, 1)
    
    # Compute derivatives
    dudt = torch.autograd.grad(u, t, grad_outputs= torch.ones_like(u), create_graph=True)[0]
    dudx = torch.autograd.grad(u, x, grad_outputs= torch.ones_like(u), create_graph=True)[0]
    d2udx2 = torch.autograd.grad(dudx, t, grad_outputs= torch.ones_like(u), create_graph= True)[0]

    # Compute residual
    return dudt + u*dudx - (0.01/np.pi()) * d2udx2

def boundary(model, tbc): # tbc size: (nbc, 1)

    # Evaluate model
    ybc = model(tbc)

    # Compute derivatives
    dydt = torch.autograd.grad(ybc, tbc, grad_outputs= torch.ones_like(ybc), create_graph=True)[0]

    x1 = ybc - 1
    x2 = dydt

    # Compute residuals
    return x1, x2

if __name__ == "__main__":

    m = 1; mu = 4; k = 400
    params = (m, mu, k)

    hlayers = 4; width = 32
    model = VanillaNetwork(hlayers, width)