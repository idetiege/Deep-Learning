import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import qmc

class PINN(nn.Module):
    def __init__(self, hlayers, width):
        super(PINN, self).__init__()
        activation = nn.Tanh

        layers = []
        layers.append(nn.Linear(2, width))
        layers.append(nn.Tanh())

        for _ in range(hlayers-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))

        self.network = nn.Sequential(*layers)
        
    def forward(self, t, x): # t is size 1,2
        return self.network(torch.cat([t, x], dim=1))
    
def residual(model, params): # t size: (ncol, 2)

    # Unpack parameters
    t, x = params

    # Evaluate model
    u = model(t, x) # y size: (ncol, 2)
    
    # Compute derivatives
    dudt = torch.autograd.grad(u, t, grad_outputs= torch.ones_like(u), create_graph=True)[0]
    dudx = torch.autograd.grad(u, x, grad_outputs= torch.ones_like(u), create_graph=True)[0]
    d2udx2 = torch.autograd.grad(dudx, x, grad_outputs= torch.ones_like(u), create_graph= True)[0]

    # Compute residual
    return dudt + u*dudx - (0.01/torch.pi) * d2udx2

def inputs(n_coll):

    # Define boundary conditions
    t0 = torch.zeros((50, 1))
    x0 = -1 + 2*torch.rand((50, 1))
    U1 = -torch.sin(np.pi * x0)
    T1 = torch.cat((t0, x0), dim = 1)

    t1 = torch.rand((25, 1))
    x1 = -torch.ones((25, 1))
    U2 = torch.zeros((25, 1))
    T2 = torch.cat((t1, x1), dim = 1)

    t2 = torch.rand((25, 1))
    x2 = torch.ones((25, 1))
    U3 = torch.zeros((25, 1))
    T3 = torch.cat((t2, x2), dim = 1)

    Xbc = torch.cat((T1, T2, T3), dim = 0)
    Ubc = torch.cat((U1, U2, U3), dim = 0)

    lower_bounds = [0, -1]
    upper_bounds = [1, 1]
    sampler = qmc.LatinHypercube(d=2)

    coll = qmc.scale(sampler.random(n_coll), lower_bounds, upper_bounds)
    coll = torch.tensor(coll, dtype = torch.float32)

    t_func = coll[:, 0:1]
    x_func = coll[:, 1:2]

    return Xbc, Ubc, t_func, x_func

def loss(func_residuals, boundary_residuals, ubc):

    MSEu = torch.mean((boundary_residuals - ubc)**2)
    MSEf = torch.mean((func_residuals)**2)
    Loss = MSEu + MSEf

    return Loss

def train(data, model, loss_fn, optimizer, epochs):

    model.train()

    Xbc, coll = data
    func_residuals = residual(model, coll)
    boundary_residuals = Xbc
    train_loss = 0
    
    for i in range(epochs):
        pred = model(t, x)
        loss = loss_fn(func_residuals, boundary_residuals, pred)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    return train_loss

if __name__ == "__main__":

    model = PINN()
    loss_fn = loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    for e in range(epochs):
        trains_loss = train()