import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import qmc

class VanillaNetwork(nn.Module):
    def __init__(self, hlayers, width):
        super(VanillaNetwork, self).__init__()
        activation = nn.Tanh

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
    
def residual(model, params): # t size: (ncol, 1)

    # Unpack parameters
    x, t = params

    # Evaluate model
    u = model(t) # y size: (ncol, 1)
    
    # Compute derivatives
    dudt = torch.autograd.grad(u, t, grad_outputs= torch.ones_like(u), create_graph=True)[0]
    dudx = torch.autograd.grad(u, x, grad_outputs= torch.ones_like(u), create_graph=True)[0]
    d2udx2 = torch.autograd.grad(dudx, x, grad_outputs= torch.ones_like(u), create_graph= True)[0]

    # Compute residual
    return dudt + u*dudx - (0.01/np.pi()) * d2udx2

def inputs(n_coll):

    # Define boundary conditions
    x0 = torch.zeros((50, 1))
    p0 = torch.rand((50, 1)) * np.pi
    g1 = -torch.sin(np.pi * p0)
    X = torch.cat((x0, g1), dim = 1)

    x1 = torch.rand((25, 1))
    g2 = -torch.ones((25, 1))
    X2 = torch.cat((x1, g2), dim = 1)

    x2 = torch.rand((25, 1))
    g3 = torch.ones((25, 1))
    X3 = torch.cat((x2, g3), dim = 1)

    Xbc = torch.cat((X, X2, X3), dim = 0)

    lower_bounds = [0, -1]
    upper_bounds = [1, 1]
    sampler = qmc.LatinHypercube(d=2)

    coll = qmc.scale(sampler.random(n_coll), lower_bounds, upper_bounds)

    return Xbc, coll

def loss(func_residuals, boundary_residuals, Nu, Nf, ubc):

    MSEu = (1/Nu) * np.sum((boundary_residuals - ubc)**2)
    MSEf = (1/Nf) * np.sum((func_residuals)**2)
    Loss = MSEu + MSEf

    return Loss



if __name__ == "__main__":

    