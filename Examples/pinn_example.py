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
    m, mu, k = params

    # Evaluate model
    y = model(t) # y size: (ncol, 1)
    
    # Compute derivatives
    dydt = torch.autograd.grad(y, t, grad_outputs= torch.ones_like(y), create_graph=True)[0]
    d2ydt2 = torch.autograd.grad(dydt, t, grad_outputs= torch.ones_like(y), create_graph= True)[0]

    # Compute residual
    return m * d2ydt2 + mu* dydt + k*y

def boundary(model, tbc): # tbc size: (nbc, 1)

    # Evaluate model
    ybc = model(tbc)

    # Compute derivatives
    dydt = torch.autograd.grad(ybc, tbc, grad_outputs= torch.ones_like(ybc), create_graph=True)[0]

    x1 = ybc - 1
    x2 = dydt

    # Compute residuals
    return x1, x2

def datapoints():
    tdata = torch.zeros(1, 1, requires_grad=True)

    return tdata

def train(tbc, tcoll, params, model, lossfn, optimizer):

    model.train()

    # Zero gradients
    optimizer.zero_grad()

    # Compute residuals
    bc1, bc2 = boundary(model, tbc)
    lossbc1 = torch.mean(bc1**2)
    lossbc2 = torch.mean(bc2**2)

    rcol = residual(model, tcoll, params)
    loss_col = torch.mean(rcol**2)

    lambda1 = 1e-4 

    # Compute loss
    loss_r = lossfn(r, torch.zeros_like(r))
    loss_b1 = lossfn(x1, torch.zeros_like(x1))
    loss_b2 = lossfn(x2, torch.zeros_like(x2))

    loss = loss_r + loss_b1 + loss_b2

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    return loss.item()

if __name__ == "__main__":

    m = 1; mu = 4; k = 400
    params = (m, mu, k)

    hlayers = 4; width = 32
    model = VanillaNetwork(hlayers, width)